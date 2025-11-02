import numpy as np

def role_assignment(teammate_positions, formation_positions):
    # Guard against None inputs
    if formation_positions is None or len(formation_positions) == 0:
        return {}
    
    if teammate_positions is None:
        return {}
    
    # Safely convert formation positions first to compute fallback center
    roles = []
    for r in formation_positions:
        if r is not None:
            roles.append(np.asarray(r, dtype=np.float64))
        else:
            # If formation has None, use fallback
            roles.append(np.array([0.0, 0.0], dtype=np.float64))
    
    # Compute formation center as fallback for None teammate positions
    if len(roles) > 0:
        formation_center = np.mean(roles, axis=0)
    else:
        formation_center = np.array([0.0, 0.0], dtype=np.float64)
    
    # Safely convert teammate positions, using formation center as fallback for None values
    # This handles cases where Strategy.py returns None for unknown teammates
    players = []
    for p in teammate_positions:
        if p is not None:
            try:
                players.append(np.asarray(p, dtype=np.float64))
            except (TypeError, ValueError):
                # If conversion fails, use fallback
                players.append(formation_center.copy())
        else:
            # Fallback: use formation center for unknown teammates
            players.append(formation_center.copy())
    
    n = len(players)
    
    # Guard against length mismatches
    if n != len(roles):
        # If mismatch, pad with fallback positions or truncate
        if n < len(roles):
            # Not enough players - pad with formation center
            while len(players) < len(roles):
                players.append(formation_center.copy())
            n = len(players)
        else:
            # Too many players - truncate to match formation size
            players = players[:len(roles)]
            n = len(players)

    def dist(i, j):
        return float(np.linalg.norm(players[i].astype(float) - roles[j].astype(float)))

    player_prefs = []
    for i in range(n):
        order = sorted(range(n), key=lambda j: (dist(i, j), j))
        player_prefs.append(order)

    role_prefs = []
    for j in range(n):
        order = sorted(range(n), key=lambda i: (dist(i, j), i))
        role_prefs.append(order)

    role_rank = [[0] * n for _ in range(n)]
    for j in range(n):
        for r, i in enumerate(role_prefs[j]):
            role_rank[j][i] = r

    role_match = [None] * n
    next_choice = [0] * n
    free_players = list(range(n))

    while free_players:
        i = free_players.pop()
        if next_choice[i] >= n:
            continue
        j = player_prefs[i][next_choice[i]]
        next_choice[i] += 1
        current = role_match[j]
        if current is None or role_rank[j][i] < role_rank[j][current]:
            if current is not None:
                free_players.append(current)
            role_match[j] = i
        else:
            free_players.append(i)

    point_preferences = {}
    for role_index, player_index in enumerate(role_match):
        # Guard against invalid indices
        if player_index is not None and 0 <= player_index < n and 0 <= role_index < len(formation_positions):
            try:
                point_preferences[player_index + 1] = formation_positions[role_index]
            except (IndexError, TypeError):
                # Skip invalid assignment
                continue

    return point_preferences
