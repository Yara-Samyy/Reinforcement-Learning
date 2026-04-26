import tkinter as tk
from tkinter import ttk
import random
import time

######----- SETUP -----######

ROWS, COLS= 5,5  ## grid dimentions

ACTIONS= {   ## values are put that way to satisfy the calculations on the grid : dr,dc
    "Up":(-1,0),
    "Down":(1,0),
    "Right":(0,1),
    "Left":(0,-1)
}

ACTION_ORDER= ["Up","Down","Right","Left"]   ## fixed order for tie-breaking

TRANSITION_PROBS= {     ## Transition probabilities
    "Up": { "Up":0.7 , "Down":0.1 , "Left":0.1 , "Right":0.1 },
    "Down": { "Down":0.7 , "Up":0.1 , "Left":0.1 , "Right":0.1 },
    "Right": { "Right":0.7 , "Left":0.1 , "Down":0.1 , "Up":0.1 },
    "Left": { "Left":0.7 , "Right":0.1 , "Down":0.1 , "Up":0.1 }
}

def put_rewards(R1,R2):  ## puting rewards on a grid
    grid= [
        [R1,1 ,0 ,-1 ,R2],
        [2 ,1 ,0 ,-1 ,-2],
        [2 ,1 ,0 ,-1 ,-2],
        [2 ,1 ,0 ,-1 ,-2],
        [2 ,1 ,0 ,-1 ,-2]
    ]
    return grid

def in_bounds(r,c):
    return  0<=r<ROWS and  0<=c<COLS

def get_next_state(state,action):
    r,c = state
    dr,dc = ACTIONS[action]
    nr,nc = r+dr , c+dc
    if in_bounds(nr,nc):
        return (nr,nc)
    return (r,c)     ## A collision with a wall results in no movement. Returning to same state

TERMINAL_STATES = [(0,0), (0,4)]  # R1 at (0,0), R2 at (0,4)

def value_iteration(rewards, gamma=0.95, theta=1e-3, max_iters=10_000, GUI=None):
    states = [(r, c) for r in range(ROWS) for c in range(COLS)]
    V = {(r, c): rewards[r][c] for r in range(ROWS) for c in range(COLS)}
    policy = {s: ACTION_ORDER[0] for s in states}

    for it in range(max_iters):
        delta = 0.0 #track max change in V this iteration
        new_V = V.copy()  # synchronous update

        for s in states:
            r, c = s

            if s in TERMINAL_STATES:
                new_V[s] = rewards[r][c]   # access reward directly
                policy[s] = None
                continue

            best_value = float("-inf") #largest q-value for this state
            best_action = None #action corresponding to best value

            for a in ACTION_ORDER:
                expected_future = 0.0

                for a2, prob in TRANSITION_PROBS[a].items():
                    s_next = get_next_state(s, a2)
                    expected_future += prob * V[s_next]

                q_sa = rewards[r][c] + gamma * expected_future  # use rewards directly

                if q_sa > best_value:
                    best_value = q_sa
                    best_action = a

            new_V[s] = best_value
            policy[s] = best_action
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V

        if GUI:
            GUI(it, V.copy(), policy.copy())

        if delta < theta:
            break

    return V, policy


# Policy Iteration (bonus)

def policy_evaluation(policy, rewards, gamma=0.95, theta=1e-3, max_iters=10_000):
    """
    Evaluate a given policy π to compute its value function V(s)
    using iterative policy evaluation.
    """

    # List of all possible states in the grid
    states = [(r, c) for r in range(ROWS) for c in range(COLS)]

    # Initialize value function V(s) = reward of each state
    V = {(r, c): rewards[r][c] for r in range(ROWS) for c in range(COLS)}

    # Iterative evaluation loop
    for _ in range(max_iters):
        delta = 0.0  # Track maximum change for convergence
        new_V = V.copy()  # Synchronous update (avoid using updated V in same iteration)

        for s in states:
            r, c = s

            # Skip terminal states as their value = the reward
            if s in TERMINAL_STATES:
                new_V[s] = rewards[r][c]
                continue

            # Get the action corresponding to the current policy
            a = policy[s]
            expected_value = 0.0

            # Compute expected value of next states under stochastic transitions
            for a2, prob in TRANSITION_PROBS[a].items():
                s_next = get_next_state(s, a2)
                expected_value += prob * V[s_next]  # Sum P(s'|s,a) * V(s')

            # Update V(s) using Bellman expectation equation
            new_V[s] = rewards[r][c] + gamma * expected_value

            # Update delta to check for convergence
            delta = max(delta, abs(new_V[s] - V[s]))

        # Apply synchronous update
        V = new_V

        # Convergence check: stop if max change < theta
        if delta < theta:
            break

    return V  # Return the value function for the given policy


def policy_improvement(V, rewards, gamma=0.95):
    """
    Improve the current policy with respect to value function V(s).
    Returns a new policy π'.
    """
    states = [(r, c) for r in range(ROWS) for c in range(COLS)]
    policy = {}  # New improved policy

    # Loop over all states to assign best action
    for s in states:

        # Terminal states do not need any action
        if s in TERMINAL_STATES:
            policy[s] = None
            continue

        best_action = None
        best_value = float('-inf')  # Initialize best value for comparison

        # Try all possible actions to find the greedy one
        for a in ACTION_ORDER:
            q_value = 0.0

            # Expected value of taking action a in state s
            for a2, prob in TRANSITION_PROBS[a].items():
                s_next = get_next_state(s, a2)
                q_value += prob * V[s_next]  # Sum P(s'|s,a) * V(s')

            # Add immediate reward of current state
            q_value = rewards[s[0]][s[1]] + gamma * q_value

            # Update best action
            if q_value > best_value:
                best_value = q_value
                best_action = a

        policy[s] = best_action # Assigned to new policy

    return policy  # Improved policy


def policy_iteration(rewards, gamma=0.95, theta=1e-3, GUI=None):
    """
    Main Policy Iteration algorithm:
    1. Initialize random policy
    2. Repeat:
        a) Policy Evaluation
        b) Policy Improvement
    Until policy is stable
    """

    states = [(r, c) for r in range(ROWS) for c in range(COLS)]

    # Initialize a random policy for non-terminal states
    policy = {
        s: random.choice(ACTION_ORDER) if s not in TERMINAL_STATES else None
        for s in states
    }

    iteration = 0  # Count number of policy improvement steps

    # Repeat until policy converges
    while True:
        # Evaluate current policy
        V = policy_evaluation(policy, rewards, gamma=gamma, theta=theta)

        # Improve policy
        new_policy = policy_improvement(V, rewards, gamma=gamma)

        # Optional GUI update to visualize current policy
        if GUI:
            GUI(iteration, V.copy(), policy.copy())

        iteration += 1  # Increment iteration counter

        # Check if policy is stable (no changes)
        if all(new_policy[s] == policy[s] for s in states):
            # Final GUI update
            if GUI:
                GUI(iteration, V.copy(), new_policy.copy())

            # Return final value function, policy, and number of iterations
            return V, new_policy, iteration

        # Update policy and repeat
        policy = new_policy


#####----- GUI -----#####

CELL_SIZE = 80  # visual size per cell, small grid so this is fine
PADDING = 10
ARROWS = {"Up": "↑", "Down": "↓", "Left": "←", "Right": "→"}

CASES = [
    ("R1=100, R2=110", (100, 110)),
    ("R1=10, R2=100",  (10, 100)),
    ("R1=1, R2=10",    (1, 10)),
    ("R1=10, R2=15",   (10, 15)),
]

class GridWorldApp:
    def __init__(self, root):
        self.root = root
        self.root.title("5x5 GridWorld — Value Iteration & Policy Iteration (γ=0.95)")
        self.gamma = 0.95
        self.theta = 1e-3   ## convergence threshold
        self.last_iteration = 0  # initialize iteration counter
        # State
        self.case_label_var = tk.StringVar(value=CASES[0][0])
        self.R1, self.R2 = CASES[0][1]
        self.rewards = put_rewards(self.R1, self.R2)
        self.terminal_states = TERMINAL_STATES

        # Initialize V to rewards instead of 0
        self.V = {(r, c): self.rewards[r][c] for r in range(ROWS) for c in range(COLS)}
        self.policy = {(r, c): "Up" for r in range(ROWS) for c in range(COLS)}
        self.show_values = tk.BooleanVar(value=True)

        # UI layout
        self._build_controls()
        self._build_canvas()
        self._draw()

    def _build_controls(self):
        # Top frame for controls
        frame = tk.Frame(self.root)
        frame.pack(pady=5)

        # Dropdown for reward cases
        tk.Label(frame, text="Select Case:").pack(side=tk.LEFT)
        self.case_combo = ttk.Combobox(frame, values=[c[0] for c in CASES], state="readonly")
        self.case_combo.pack(side=tk.LEFT)
        self.case_combo.current(0)
        self.case_combo.bind("<<ComboboxSelected>>", self.on_case_change)

        # Buttons
        tk.Button(frame, text="Run Value Iteration", command=self.run_value_iteration).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Run Policy Iteration", command=self.run_policy_iteration).pack(side=tk.LEFT, padx=5)

        # Checkbox to show/hide values
        tk.Checkbutton(frame, text="Show Values", variable=self.show_values, command=self._draw).pack(side=tk.LEFT, padx=5)

        # Info label
        self.info_var = tk.StringVar(value="Select a case and run an algorithm")
        tk.Label(self.root, textvariable=self.info_var).pack(pady=2)

    def _build_canvas(self):
        self.canvas = tk.Canvas(
            self.root,
            width=COLS * CELL_SIZE + 2 * PADDING,
            height=ROWS * CELL_SIZE + 2 * PADDING,
            bg="white"
        )
        self.canvas.pack(pady=5)


    def on_case_change(self, _evt=None):
        selected = self.case_combo.get()
        for label, (R1, R2) in CASES:
            if label == selected:
                self.R1, self.R2 = R1, R2
                break
        self.rewards = put_rewards(self.R1, self.R2)
        # Reset values and policy using rewards
        self.V = {(r, c): self.rewards[r][c] for r in range(ROWS) for c in range(COLS)}
        self.policy = {(r, c): "Up" for r in range(ROWS) for c in range(COLS)}
        self.info_var.set(f"Case changed: {selected}")
        self._draw()

    def run_value_iteration(self):
        self.V, self.policy = value_iteration(self.rewards, gamma=self.gamma, theta=self.theta,GUI=self.gui_update)
        self.info_var.set(f"Value Iteration: converged in {self.last_iteration} iterations")
        self._draw()

    def run_policy_iteration(self):
        self.V, self.policy, self.last_iteration = policy_iteration(self.rewards, gamma=self.gamma, theta=self.theta, GUI=self.gui_update)
        self.info_var.set(f"Policy Iteration: converged in {self.last_iteration} iterations")
        self._draw()

    def gui_update(self, iteration, V, policy):
        self.V = V
        self.policy = policy
        self.last_iteration = iteration
        self.info_var.set(f"Iteration {iteration}")
        self._draw()
        # Force Tkinter to update immediately
        self.root.update_idletasks()
        self.root.update()
        time.sleep(0.05)  ## added delay


    def _draw_cell(self, r, c):
        x0 = PADDING + c * CELL_SIZE
        y0 = PADDING + r * CELL_SIZE
        x1 = x0 + CELL_SIZE
        y1 = y0 + CELL_SIZE

        # Background color based on reward (visual cue, lightweight)
        reward = self.rewards[r][c]
        if reward > 0:
            fill = "#e8fff0"  # light green
        elif reward < 0:
            fill = "#ffecec"  # light red
        else:
            fill = "#f5f5f5"  # light gray

        self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#cccccc")

        # Reward top-left
        self.canvas.create_text(x0 + 8, y0 + 10, text=f"R={reward}", anchor="nw", fill="#555", font=("Segoe UI", 9))

        # Value bottom-left
        if self.show_values.get():
            val = self.V[(r, c)]
            self.canvas.create_text(x0 + 8, y1 - 10, text=f"V={val:.2f}", anchor="sw", fill="#222", font=("Segoe UI", 10))

        # Policy arrow centered
        #action = self.policy[(r, c)]
        #arrow = ARROWS.get(action, "·")
        # Terminal state: draw T instead of arrow
        if (r, c) in TERMINAL_STATES:
            arrow = "T"
        else:
            arrow = ARROWS.get(self.policy[(r, c)], "·")

        self.canvas.create_text((x0 + x1) // 2, (y0 + y1) // 2, text=arrow, fill="#000", font=("Segoe UI", 16))

    def _draw_grid_lines(self):
        # Optional grid lines for clarity
        for r in range(ROWS + 1):
            y = PADDING + r * CELL_SIZE
            self.canvas.create_line(PADDING, y, PADDING + COLS * CELL_SIZE, y, fill="#dddddd")
        for c in range(COLS + 1):
            x = PADDING + c * CELL_SIZE
            self.canvas.create_line(x, PADDING, x, PADDING + ROWS * CELL_SIZE, fill="#dddddd")

    def _draw(self):
        self.canvas.delete("all")
        self._draw_grid_lines()
        for r in range(ROWS):
            for c in range(COLS):
                self._draw_cell(r, c)
                

if __name__ == "__main__":
    # Launch GUI
    root = tk.Tk()
    GridWorldApp(root)
    root.mainloop()
    