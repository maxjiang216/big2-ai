# Sample games (anomalies)

From this run: **longest** / **shortest** game by turn count; **most** / **fewest** legal moves for the opening player before the first card (same as feature `start_legal_moves`).

`game_index` is the global game number for this run (0 … N−1), stable after sorting regardless of `--threads`.

## Game `32085` — *longest game*

- **Turns:** 50
- **Winner:** P1
- **Opening legal moves (start_legal_moves):** 27

### Starting hands (before move 0)


| Player | Cards (rank symbols; 3 lowest … 2 highest) |
| ------ | ------------------------------------------ |
| P0     | `344566789JQQKAA2`                         |
| P1     | `3446777888990JJK`                         |


### Turn-by-turn history


| Turn | Player to move | Move   | `move_id` |
| ---- | -------------- | ------ | --------- |
| 0    | P0             | `QQ`   | 23        |
| 1    | P1             | `PASS` | 0         |
| 2    | P0             | `7`    | 5         |
| 3    | P1             | `8`    | 6         |
| 4    | P0             | `A`    | 12        |
| 5    | P1             | `PASS` | 0         |
| 6    | P0             | `66`   | 17        |
| 7    | P1             | `PASS` | 0         |
| 8    | P0             | `2`    | 13        |
| 9    | P1             | `PASS` | 0         |
| 10   | P0             | `J`    | 9         |
| 11   | P1             | `PASS` | 0         |
| 12   | P0             | `9`    | 7         |
| 13   | P1             | `J`    | 9         |
| 14   | P0             | `PASS` | 0         |
| 15   | P1             | `8`    | 6         |
| 16   | P0             | `PASS` | 0         |
| 17   | P1             | `K`    | 11        |
| 18   | P0             | `PASS` | 0         |
| 19   | P1             | `7`    | 5         |
| 20   | P0             | `PASS` | 0         |
| 21   | P1             | `0`    | 8         |
| 22   | P0             | `A`    | 12        |
| 23   | P1             | `PASS` | 0         |
| 24   | P0             | `K`    | 11        |
| 25   | P1             | `PASS` | 0         |
| 26   | P0             | `5`    | 3         |
| 27   | P1             | `PASS` | 0         |
| 28   | P0             | `4`    | 2         |
| 29   | P1             | `9`    | 7         |
| 30   | P0             | `PASS` | 0         |
| 31   | P1             | `7`    | 5         |
| 32   | P0             | `PASS` | 0         |
| 33   | P1             | `3`    | 1         |
| 34   | P0             | `8`    | 6         |
| 35   | P1             | `PASS` | 0         |
| 36   | P0             | `3`    | 1         |
| 37   | P1             | `4`    | 2         |
| 38   | P0             | `PASS` | 0         |
| 39   | P1             | `J`    | 9         |
| 40   | P0             | `PASS` | 0         |
| 41   | P1             | `9`    | 7         |
| 42   | P0             | `PASS` | 0         |
| 43   | P1             | `8`    | 6         |
| 44   | P0             | `PASS` | 0         |
| 45   | P1             | `7`    | 5         |
| 46   | P0             | `PASS` | 0         |
| 47   | P1             | `6`    | 4         |
| 48   | P0             | `PASS` | 0         |
| 49   | P1             | `4`    | 2         |


## Game `5685` — *shortest game*

- **Turns:** 3
- **Winner:** P0
- **Opening legal moves (start_legal_moves):** 64

### Starting hands (before move 0)


| Player | Cards (rank symbols; 3 lowest … 2 highest) |
| ------ | ------------------------------------------ |
| P0     | `34567777890JQKAA`                         |
| P1     | `334445888990JQK2`                         |


### Turn-by-turn history


| Turn | Player to move | Move          | `move_id` |
| ---- | -------------- | ------------- | --------- |
| 0    | P0             | `34567890JQK` | 371       |
| 1    | P1             | `PASS`        | 0         |
| 2    | P0             | `777AA`       | 91        |


## Game `25978` — *most opening legal moves*

- **Turns:** 5
- **Winner:** P0
- **Opening legal moves (start_legal_moves):** 83

### Starting hands (before move 0)


| Player | Cards (rank symbols; 3 lowest … 2 highest) |
| ------ | ------------------------------------------ |
| P0     | `34567890JQKKAAA2`                         |
| P1     | `33444556890JJQQK`                         |


### Turn-by-turn history


| Turn | Player to move | Move            | `move_id` |
| ---- | -------------- | --------------- | --------- |
| 0    | P0             | `34567890JQKA2` | 377       |
| 1    | P1             | `PASS`          | 0         |
| 2    | P0             | `AA`            | 25        |
| 3    | P1             | `PASS`          | 0         |
| 4    | P0             | `K`             | 11        |


## Game `3133` — *fewest opening legal moves*

- **Turns:** 37
- **Winner:** P0
- **Opening legal moves (start_legal_moves):** 16

### Starting hands (before move 0)


| Player | Cards (rank symbols; 3 lowest … 2 highest) |
| ------ | ------------------------------------------ |
| P0     | `33456688900QQKAA`                         |
| P1     | `345667780JJQQKA2`                         |


### Turn-by-turn history


| Turn | Player to move | Move   | `move_id` |
| ---- | -------------- | ------ | --------- |
| 0    | P0             | `5`    | 3         |
| 1    | P1             | `K`    | 11        |
| 2    | P0             | `A`    | 12        |
| 3    | P1             | `PASS` | 0         |
| 4    | P0             | `00`   | 21        |
| 5    | P1             | `QQ`   | 23        |
| 6    | P0             | `PASS` | 0         |
| 7    | P1             | `77`   | 18        |
| 8    | P0             | `QQ`   | 23        |
| 9    | P1             | `PASS` | 0         |
| 10   | P0             | `33`   | 14        |
| 11   | P1             | `JJ`   | 22        |
| 12   | P0             | `PASS` | 0         |
| 13   | P1             | `8`    | 6         |
| 14   | P0             | `A`    | 12        |
| 15   | P1             | `2`    | 13        |
| 16   | P0             | `PASS` | 0         |
| 17   | P1             | `66`   | 17        |
| 18   | P0             | `PASS` | 0         |
| 19   | P1             | `4`    | 2         |
| 20   | P0             | `PASS` | 0         |
| 21   | P1             | `A`    | 12        |
| 22   | P0             | `PASS` | 0         |
| 23   | P1             | `5`    | 3         |
| 24   | P0             | `6`    | 4         |
| 25   | P1             | `PASS` | 0         |
| 26   | P0             | `8`    | 6         |
| 27   | P1             | `PASS` | 0         |
| 28   | P0             | `8`    | 6         |
| 29   | P1             | `PASS` | 0         |
| 30   | P0             | `9`    | 7         |
| 31   | P1             | `0`    | 8         |
| 32   | P0             | `K`    | 11        |
| 33   | P1             | `PASS` | 0         |
| 34   | P0             | `6`    | 4         |
| 35   | P1             | `PASS` | 0         |
| 36   | P0             | `4`    | 2         |


