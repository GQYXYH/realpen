# User Configuration

## Usage of this directory
The files in this directory are not allowed to be updated between different users of the repository since they are system dependend configurations (excluding this ReadMe). I.e. if you set global variables which can be different for users, then these can be defined here.

After cloning this repository, make sure that your files in this folder will not be tracked anymore by using the `update-index` command in the repository root folder:
```git
$ git update-index --skip-worktree FILENAME
```

### Example
For the `configuration.py` file this looks as followed:
```git
$ git update-index --skip-worktree gym_brt/data/config/configuration.py
```