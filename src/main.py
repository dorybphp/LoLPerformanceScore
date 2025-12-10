import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_pipeline import load_and_preprocess_data, build_dataloaders
from model import PlayerImpactModel
from utils import set_seed, save_checkpoint, load_checkpoint, plot_history, ensure_dir

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    preds=[]; trues=[]
    for batch in tqdm(train_loader, desc="train", leave=False):
        player, champ, role, team, league, tourn, series, numagg, y = [b.to(device) for b in batch]
        optimizer.zero_grad()
        logits, score, z, pvec = model(player, champ, role, team, league, tourn, series, numagg)
        loss = criterion(logits, y.squeeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * player.size(0)
        preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        trues.append(y.cpu().numpy())
    return running_loss / len(train_loader.dataset), np.concatenate(preds), np.concatenate(trues)


def evaluate(model, criterion, loader, device):
    model.eval()
    loss = 0.0
    preds=[]; trues=[]
    with torch.no_grad():
        for batch in loader:
            player, champ, role, team, league, tourn, series, numagg, y = [b.to(device) for b in batch]
            logits, score, z, pvec = model(player, champ, role, team, league, tourn, series, numagg)
            loss += criterion(logits, y.squeeze(1)).item() * player.size(0)
            preds.append(torch.sigmoid(logits).cpu().numpy())
            trues.append(y.cpu().numpy())
    return loss / len(loader.dataset), np.concatenate(preds), np.concatenate(trues)


def main():
    set_seed(42)
    ensure_dir('checkpoints')
    ensure_dir('results')

    # load and preprocess
    (X_player, X_champ, X_role, X_team, X_league, X_tourn, X_series, X_numagg), \
        y, df, encoders, scaler = load_and_preprocess_data(stats_csv='game_players_stats.csv', meta_csv='game_metadata.csv')

    # build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders((X_player, X_champ, X_role, X_team, X_league, X_tourn, X_series, X_numagg), y)

    # model init
    num_players = int(df['player_id_idx'].nunique())
    num_champs  = int(df['champion_name_idx'].nunique())
    num_roles   = int(df['role_idx'].nunique())
    num_teams   = int(df['team_id_idx'].nunique())
    num_leagues = int(df['league_name_idx'].nunique())
    num_tourns  = int(df['tournament_name_idx'].nunique())
    num_series  = int(df['series_name_idx'].nunique())

    model = PlayerImpactModel(
        n_players=num_players, n_champs=num_champs, n_roles=num_roles, n_teams=num_teams,
        n_leagues=num_leagues, n_tourns=num_tourns, n_series=num_series,
        numagg_dim=X_numagg.shape[1]
    ).to(DEVICE)

    # loss, optimizer, scheduler
    train_labels = y[df['train_mask']]
    pos = float(train_labels.sum()); neg = float(len(train_labels) - pos)
    pos_weight = (neg / (pos + 1e-9)) if pos>0 else 1.0
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    EPOCHS = 12
    best_val_auc = -np.inf
    no_improve = 0
    early_stop_patience = 4

    history = {'train_loss':[], 'val_loss':[], 'train_auc':[], 'val_auc':[]}

    for epoch in range(1, EPOCHS+1):
        train_loss, train_preds, train_trues = train(model, optimizer, criterion, train_loader, DEVICE)
        val_loss, val_preds, val_trues = evaluate(model, criterion, val_loader, DEVICE)

        # compute simple AUC where possible
        try:
            from sklearn.metrics import roc_auc_score
            train_auc = roc_auc_score(train_trues, train_preds) if len(np.unique(train_trues))>1 else float('nan')
            val_auc   = roc_auc_score(val_trues, val_preds) if len(np.unique(val_trues))>1 else float('nan')
        except Exception:
            train_auc = val_auc = float('nan')

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc); history['val_auc'].append(val_auc)

        print(f"Epoch {epoch}/{EPOCHS} train_loss={train_loss:.4f} val_loss={val_loss:.4f} train_auc={train_auc:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_val_auc and not np.isnan(val_auc):
            best_val_auc = val_auc
            save_checkpoint(model, encoders, 'checkpoints/model.pt')
            no_improve = 0
            print(f"Saved best model at epoch {epoch} val_auc {val_auc:.4f}")
        else:
            no_improve += 1

        if no_improve >= early_stop_patience:
            print("Early stopping")
            break

    # load best checkpoint for final evaluation
    if os.path.exists('checkpoints/model.pt'):
        ck = load_checkpoint(model, 'checkpoints/model.pt', DEVICE)
        print('Loaded best checkpoint')

    # final evaluation on test set + produce CSVs
    model.eval()
    all_probs=[]; all_scores=[]; all_labels=[]
    from torch.utils.data import DataLoader
    full_ds = None
    # build full dataset for scoring and leaderboard
    from data_pipeline import build_full_dataset
    full_ds = build_full_dataset((X_player, X_champ, X_role, X_team, X_league, X_tourn, X_series, X_numagg), y)
    full_loader = DataLoader(full_ds, batch_size=512)

    rows_preds=[]; rows_scores=[]; rows_pvecs=[]
    with torch.no_grad():
        for batch in full_loader:
            player, champ, role, team, league, tourn, series, numagg, yb = [b.to(DEVICE) for b in batch]
            logits, score, z, pvec = model(player, champ, role, team, league, tourn, series, numagg)
            probs = torch.sigmoid(logits).cpu().numpy()
            rows_preds.append(probs.reshape(-1))
            rows_scores.append(score.cpu().numpy().reshape(-1))
            rows_pvecs.append(pvec.cpu().numpy())

    import numpy as np
    rows_preds = np.concatenate(rows_preds)
    rows_scores = np.concatenate(rows_scores)
    rows_pvecs = np.vstack(rows_pvecs)

    # attach back to df and save CSVs
    df = df.reset_index(drop=True)
    df['pred'] = rows_preds
    df['hidden_score_raw'] = rows_scores
    df['player_emb_row'] = list(rows_pvecs)

    # scale hidden score to 0-100
    score_scaled = (df['hidden_score_raw'] - df['hidden_score_raw'].min()) / (df['hidden_score_raw'].max() - df['hidden_score_raw'].min() + 1e-9)
    df['score_scaled'] = score_scaled * 100

    score_export_cols = ['game_id', 'player_id', 'player_name', 'team_id', 'team_name', 'role', 'champion_name', 'score_scaled']
    df[score_export_cols].to_csv('results/model_scores_per_row.csv', index=False)
    print('Saved: results/model_scores_per_row.csv')

    # build leaderboard
    config_agg = df.groupby(['player_name', 'champion_name', 'team_name']).agg(mean_winprob=('pred','mean'), games=('pred','count')).reset_index()
    config_agg['favorability_0_100'] = config_agg['mean_winprob'] * 100
    leaderboard = config_agg[config_agg['games']>=5].sort_values('favorability_0_100', ascending=False).head(100).reset_index(drop=True)
    leaderboard.to_csv('results/player_champion_team_leaderboard.csv', index=False)
    print('Saved: results/player_champion_team_leaderboard.csv')

    # plot training curves
    plot_history(history, outpath='results/training_history.png')

if __name__ == '__main__':
    main()
