# Usage of this directory
The files in this directory are not allowed to be updated between different users of the repository since they are system dependend configurations (excluding this ReadMe). 
After pulling from the repository, make sure that the your file will not be tracked anymore:
```git
$ git update-index --skip-worktree FILENAME
```

Example:
```git
$ git update-index --skip-worktree gym_brt/data/config/configuration.py
```