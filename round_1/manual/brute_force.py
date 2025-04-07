import numpy as np

rates = {
    "snowballs":{
        "snowballs": 1,
        "pizzas": 1.45,
        "nuggets": .52,
        "shells": .72
        },
    "pizzas":{
        "snowballs": .7,
        "pizzas": 1,
        "nuggets": .31,
        "shells": .48
    },
    "nuggets":{
        "snowballs": 1.95,
        "pizzas": 3.1,
        "nuggets": 1,
        "shells": 1.49
        },
    "shells":{
        "snowballs": 1.34,
        "pizzas": 1.98,
        "nuggets": .64,
        "shells": 1
    },
}

currencies = list(rates.keys())
trials = []

#overkill i know, but just simulates all possible traversals, valid or not
for i in range(4):
    for j in range(4):
        for k in range(4):
            for x in range(4):
                for y in range(4):
                    for z in range(4):
                        trials.append([
                            (currencies[i], 1),
                            (currencies[j], rates[currencies[i]][currencies[j]]),
                            (currencies[k], rates[currencies[i]][currencies[j]] * rates[currencies[j]][currencies[k]]),
                            (currencies[x], rates[currencies[i]][currencies[j]] * rates[currencies[j]][currencies[k]] * rates[currencies[k]][currencies[x]]),
                            (currencies[y], rates[currencies[i]][currencies[j]] * rates[currencies[j]][currencies[k]] * rates[currencies[k]][currencies[x]] * rates[currencies[x]][currencies[y]]),
                            (currencies[z], rates[currencies[i]][currencies[j]] * rates[currencies[j]][currencies[k]] * rates[currencies[k]][currencies[x]] * rates[currencies[x]][currencies[y]] * rates[currencies[y]][currencies[z]])
                        ])

best_trial = []
best_trial_profit = 0
best_shell_trial = []
best_shell_trial_profit = 0

#filter to be a cycle and better than previous
for trial in trials:
    if trial[0][0] == trial[-1][0]:
        if trial[-1][1] - 1 >= best_trial_profit:
            best_trial = trial
            best_trial_profit = trial[-1][1] - 1

        #check that cycle starts (and ends) on shells
        if trial[0][0] == "shells" and trial[-1][1] - 1 >= best_shell_trial_profit:
            best_shell_trial = trial
            best_shell_trial_profit = trial[-1][1] - 1

print(f"Best Trial {best_trial} with profit {best_trial_profit}")
print(f"Best Usable (Shell) Trial {best_shell_trial} with profit {best_shell_trial_profit}")