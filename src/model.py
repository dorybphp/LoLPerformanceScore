import torch
import torch.nn as nn

class PlayerImpactModel(nn.Module):
    def __init__(self,
                 n_players, n_champs, n_roles, n_teams, n_leagues, n_tourns, n_series,
                 numagg_dim,
                 p_emb=64, c_emb=32, r_emb=8, t_emb=16, l_emb=8, tn_emb=8, s_emb=8,
                 hidden=[256,128], z_dim=64, dropout=0.3):
        super().__init__()
        self.player_emb = nn.Embedding(n_players, p_emb)
        self.champ_emb  = nn.Embedding(n_champs, c_emb)
        self.role_emb   = nn.Embedding(n_roles, r_emb)
        self.team_emb   = nn.Embedding(n_teams, t_emb)
        self.league_emb = nn.Embedding(n_leagues, l_emb)
        self.tourn_emb  = nn.Embedding(n_tourns, tn_emb)
        self.series_emb = nn.Embedding(n_series, s_emb)

        in_dim = p_emb + r_emb + c_emb + t_emb + l_emb + tn_emb + s_emb + numagg_dim
        layers = []
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, z_dim))
        layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

        self.win_head = nn.Linear(z_dim, 1)
        self.score_head = nn.Linear(z_dim, 1)

    def forward(self, player, champ, role, team, league, tourn, series, numagg):
        p = self.player_emb(player)
        r = self.role_emb(role)
        player_vector = torch.cat([p, r], dim=1)
        c = self.champ_emb(champ)
        t = self.team_emb(team)
        l = self.league_emb(league)
        tn = self.tourn_emb(tourn)
        s = self.series_emb(series)
        x = torch.cat([player_vector, c, t, l, tn, s, numagg], dim=1)
        z = self.encoder(x)
        logits = self.win_head(z).squeeze(1)
        score = self.score_head(z).squeeze(1)
        return logits, score, z, player_vector
