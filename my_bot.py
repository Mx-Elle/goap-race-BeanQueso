from collections import deque
import numpy as np

from game_world.racetrack import RaceTrack

Point = tuple[int, int]
Action = tuple[int, int]
State = tuple[int, int, int]  # (row, col, toggle_mask) -> toggle mask tracks wall colors being flipped

MOVES: list[Action] = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def reconstruct_path(
    came_from: dict[State, tuple[State, Action]], current: State
) -> list[Action]:
    total_path: list[Action] = []
    while current in came_from:
        current, action = came_from[current]
        total_path.append(action)
    total_path.reverse()
    return total_path


def astar(
    start: State,
    goal: Point,
    neighbors_fn,
    heuristic_fn,
) -> list[Action]:
    open_set: set[State] = {start}
    came_from: dict[State, tuple[State, Action]] = {}

    g_score: dict[State, int] = {start: 0}
    f_score: dict[State, int] = {start: heuristic_fn(start)}

    while open_set:
        current = min(open_set, key=lambda s: f_score.get(s, float("inf")))
        if (current[0], current[1]) == goal:
            return reconstruct_path(came_from, current)

        open_set.remove(current)
        for neighbor, action in neighbors_fn(current):
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = (current, action)
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic_fn(neighbor)
                if neighbor not in open_set:
                    open_set.add(neighbor)

    return []


class Planner:
    def __init__(self) -> None:
        self.plan: deque[Action] = deque()
        self.expected_loc: Point | None = None

    def next_move(self, loc: Point, track: RaceTrack) -> Action:
        if not self.plan or self.expected_loc != loc:
            self.plan = deque(self._plan(loc, track))
            self.expected_loc = loc
        if not self.plan:
            return self._fallback(loc, track)
        action = self.plan.popleft()
        self.expected_loc = (loc[0] + action[0], loc[1] + action[1])
        return action

    def _plan(self, loc: Point, track: RaceTrack) -> list[Action]:
        walls = track.walls.astype(int)
        active0 = track.active.astype(int)
        buttons = track.buttons.astype(int)
        colors = track.colors.astype(int)
        rows, cols = walls.shape

        wall_color_values = np.unique(colors[walls.astype(bool)])
        color_to_bit = {
            int(color): 1 << i
            for i, color in enumerate(sorted(map(int, wall_color_values.tolist())))
        }
        color_bits = np.zeros_like(walls, dtype=int)
        for color, bit in color_to_bit.items():
            color_bits[colors == color] = bit

        start: State = (loc[0], loc[1], 0)
        target = track.target

        def heuristic(state: State) -> int:
            return abs(state[0] - target[0]) + abs(state[1] - target[1])

        def neighbors(state: State) -> list[tuple[State, Action]]:
            r, c, mask = state
            output: list[tuple[State, Action]] = []
            for dr, dc in MOVES:
                nr, nc = r + dr, c + dc
                if nr not in range(rows) or nc not in range(cols):
                    continue
                if walls[nr, nc]:
                    active = active0[nr, nc]
                    bit = color_bits[nr, nc]
                    if bit and (mask & bit):
                        active = 1 - active
                    if active:
                        continue
                nmask = mask
                if buttons[nr, nc]:
                    bit = color_bits[nr, nc]
                    if bit:
                        if nmask & bit:
                            nmask = nmask - bit
                        else:
                            nmask = nmask + bit
                output.append(((nr, nc, nmask), (dr, dc)))
            return output

        return astar(start, target, neighbors, heuristic)

    def _fallback(self, loc: Point, track: RaceTrack) -> Action:
        safe = track.find_traversable_cells()
        for dr, dc in MOVES:
            if (loc[0] + dr, loc[1] + dc) in safe:
                return (dr, dc)
        return (0, 0)


PLANNER = Planner()


def move(loc: Point, track: RaceTrack) -> Action:
    return PLANNER.next_move(loc, track)
