import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset


class PlayerDataset(Dataset):
    def __init__(self, player, champ, role, team, league, tourn, series, numagg, y):
        self.player = torch.tensor(player, dtype=torch.long)
        self.champ  = torch.tensor(champ, dtype=torch.long)
        self.role   = torch.tensor(role, dtype=torch.long)
        self.team   = torch.tensor(team, dtype=torch.long)
        self.league = torch.tensor(league, dtype=torch.long)
        self.tourn  = torch.tensor(tourn, dtype=torch.long)
        self.series = torch.tensor(series, dtype=torch.long)
        self.numagg = torch.tensor(numagg, dtype=torch.float32)
        self.y      = torch.tensor(y, dtype=torch.float32).reshape(-1,1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.player[idx], self.champ[idx], self.role[idx], self.team[idx],
                self.league[idx], self.tourn[idx], self.series[idx], self.numagg[idx], self.y[idx])


def load_and_preprocess_data(stats_csv='game_players_stats.csv', meta_csv='game_metadata.csv', game_floor=50):
    df = pd.read_csv(stats_csv)
    meta = pd.read_csv(meta_csv)

    # keep subset of meta
    meta_small = meta[['game_id','date','match_id','tournament_id','tournament_name','series_id','series_name','league_id','league_name']].copy()
    df = df.merge(meta_small, on='game_id', how='left')

    # normalize role labels
    df['role'] = df['role'].astype(str).str.strip().replace({'Bot':'ADC'})
    valid_roles = ['Top','Jungle','Mid','ADC','Support']
    df = df[df['role'].isin(valid_roles)].copy()

    # filter players with > game_floor games
    games_per_player = df.groupby('player_name').size()
    eligible_players = games_per_player[games_per_player > game_floor].index.tolist()
    df = df[df['player_name'].isin(eligible_players)].copy()

    # map win to numeric
    df['win'] = df['win'].map({True:1, False:0, 'True':1, 'False':0}).astype(float).fillna(0.0)

    # date parse and sort
    if 'date' in df.columns and df['date'].notnull().any():
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)
    else:
        df = df.sort_values('game_id').reset_index(drop=True)

    # stat columns
    stat_cols = [
        'player_kills','player_deaths','player_assists',
        'total_minions_killed','gold_earned','level',
        'total_damage_dealt','total_damage_dealt_to_champions',
        'total_damage_taken','wards_placed','largest_killing_spree','largest_multi_kill'
    ]

    for c in stat_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[stat_cols] = df[stat_cols].fillna(0.0)

    eps = 1e-9
    for c in stat_cols:
        role_means = df.groupby('role')[c].transform('mean')
        role_stds  = df.groupby('role')[c].transform(lambda x: x.std(ddof=0)).replace(0, eps)
        df[c + '_rnorm'] = (df[c] - role_means) / (role_stds + eps)

    stat_rnorm_cols = [c + '_rnorm' for c in stat_cols]

    # team aggregates
    agg_stat_cols = stat_rnorm_cols.copy()
    team_grp = df.groupby(['game_id','team_id'])
    team_means = team_grp[agg_stat_cols].mean().add_prefix('team_mean_').reset_index()
    team_sums  = team_grp[agg_stat_cols].sum().add_prefix('team_sum_').reset_index()
    team_sizes = team_grp.size().reset_index(name='team_size')

    df = df.merge(team_means, on=['game_id','team_id'], how='left')
    df = df.merge(team_sums, on=['game_id','team_id'], how='left')
    df = df.merge(team_sizes, on=['game_id','team_id'], how='left')

    for s in agg_stat_cols:
        df[f'team_ex_player_mean_{s}'] = ((df[f'team_sum_{s}'] - df[s]) / df['team_size'].replace(1, np.nan))
        df[f'team_ex_player_mean_{s}'] = df[f'team_ex_player_mean_{s}'].fillna(df[f'team_mean_{s}'])

    team_means_lookup = team_means.groupby('game_id').apply(lambda g: g.set_index('team_id').to_dict('index')).to_dict()
    def enemy_means_row(game_id, team_id):
        d = team_means_lookup.get(game_id, {})
        for t_id, row in d.items():
            if t_id != team_id:
                return {('enemy_' + k): v for k,v in row.items()}
        return {('enemy_team_mean_' + c): 0.0 for c in agg_stat_cols}

    enemy_df = df.apply(lambda r: pd.Series(enemy_means_row(r['game_id'], r['team_id'])), axis=1)
    df = pd.concat([df.reset_index(drop=True), enemy_df.reset_index(drop=True)], axis=1)
    df.fillna(0.0, inplace=True)

    team_agg_cols = [c for c in df.columns if c.startswith('team_ex_player_mean_')]
    enemy_agg_cols = [c for c in df.columns if c.startswith('enemy_') and any(s in c for s in agg_stat_cols)]
    numeric_agg = team_agg_cols + enemy_agg_cols
    if len(numeric_agg) == 0:
        df['dummy_agg'] = 0.0
        numeric_agg = ['dummy_agg']

    # categorical encoding
    cat_cols = ['player_id','player_name','champion_name','role','team_id','team_name','league_name','tournament_name','series_name']
    for c in cat_cols:
        df[c] = df[c].fillna('Unknown').astype(str)

    encoders = {}
    for c in ['player_id','champion_name','role','team_id','league_name','tournament_name','series_name']:
        le = LabelEncoder()
        df[c + '_idx'] = le.fit_transform(df[c])
        encoders[c] = le

    scaler = StandardScaler()
    df[numeric_agg] = scaler.fit_transform(df[numeric_agg].values)

    # input arrays
    X_player = df['player_id_idx'].values.astype(np.int64)
    X_champ  = df['champion_name_idx'].values.astype(np.int64)
    X_role   = df['role_idx'].values.astype(np.int64)
    X_team   = df['team_id_idx'].values.astype(np.int64)
    X_league = df['league_name_idx'].values.astype(np.int64)
    X_tourn  = df['tournament_name_idx'].values.astype(np.int64)
    X_series = df['series_name_idx'].values.astype(np.int64)
    X_numagg = df[numeric_agg].values.astype(np.float32)
    y_win    = df['win'].values.astype(np.float32)

    # time-aware split
    unique_games = np.sort(df['game_id'].unique())
    n = len(unique_games)
    train_cut = int(n * 0.7)
    val_cut   = int(n * 0.85)
    train_games = unique_games[:train_cut]
    val_games   = unique_games[train_cut:val_cut]
    test_games  = unique_games[val_cut:]

    df['train_mask'] = df['game_id'].isin(train_games)
    df['val_mask']   = df['game_id'].isin(val_games)
    df['test_mask']  = df['game_id'].isin(test_games)

    return (X_player, X_champ, X_role, X_team, X_league, X_tourn, X_series, X_numagg), y_win, df, encoders, scaler


def build_dataloaders(inputs, y, batch=256):
    X_player, X_champ, X_role, X_team, X_league, X_tourn, X_series, X_numagg = inputs
    # this function expects that train/val/test masks are present in a global df variable
    # to be simple, caller should re-run load_and_preprocess_data and capture df; we will create ds using masks from df
    raise RuntimeError("Use build_dataloaders_from_df(df, inputs, y, batch) instead")


def build_dataloaders_from_df(df, inputs, y, batch=256):
    X_player, X_champ, X_role, X_team, X_league, X_tourn, X_series, X_numagg = inputs
    train_ds = PlayerDataset(X_player[df['train_mask']], X_champ[df['train_mask']], X_role[df['train_mask']], X_team[df['train_mask']],
                             X_league[df['train_mask']], X_tourn[df['train_mask']], X_series[df['train_mask']], X_numagg[df['train_mask']], y[df['train_mask']])
    val_ds   = PlayerDataset(X_player[df['val_mask']], X_champ[df['val_mask']], X_role[df['val_mask']], X_team[df['val_mask']],
                             X_league[df['val_mask']], X_tourn[df['val_mask']], X_series[df['val_mask']], X_numagg[df['val_mask']], y[df['val_mask']])
    test_ds  = PlayerDataset(X_player[df['test_mask']], X_champ[df['test_mask']], X_role[df['test_mask']], X_team[df['test_mask']],
                             X_league[df['test_mask']], X_tourn[df['test_mask']], X_series[df['test_mask']], X_numagg[df['test_mask']], y[df['test_mask']])

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch)
    test_loader  = DataLoader(test_ds, batch_size=batch)

    return train_loader, val_loader, test_loader


def build_full_dataset(inputs, y):
    X_player, X_champ, X_role, X_team, X_league, X_tourn, X_series, X_numagg = inputs
    return PlayerDataset(X_player, X_champ, X_role, X_team, X_league, X_tourn, X_series, X_numagg, y)
