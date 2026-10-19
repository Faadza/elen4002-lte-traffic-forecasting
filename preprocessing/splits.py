# Shared chronology constants. Half-open hourly indices on the full calendar.
# The calendar runs 2017-10-23 00:00 to 2018-10-22 23:00, 8760 hours.

N_HOURS = 8760
TRAIN_END = 6132   # first 70%: [0, 6132)
VAL_END = 7446     # confirmation: [6132, 7446)
                   # final test: [7446, 8760)

# expanding development folds inside the training block
FOLDS = [
    {"train_end": 4116, "eval_start": 4116, "eval_end": 4788},
    {"train_end": 4788, "eval_start": 4788, "eval_end": 5460},
    {"train_end": 5460, "eval_start": 5460, "eval_end": 6132},
]

CALENDAR_START = "2017-10-23 00:00"

MAX_FILL_HOURS = 3
LOOKBACK = 168
