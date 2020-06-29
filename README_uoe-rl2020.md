# uoe-rl2020
Coursework base code for Reinforcement Learning 2020 (UoE)

Find the coursework description here (might need Learn log-in): 

https://www.learn.ed.ac.uk/bbcswebdav/pid-4067850-dt-content-rid-12120423_1/xid-12120423_1 

## Installation Instructions (DICE)

Open the terminal and write:

```bash
mkvirtualenv --python=`which python3` rl2020
```
This will create a virtual environment called 'rl2020' which you should use to work.

It's a python3 environment since python3 will be used during tests.

Every time you open a terminal you will need to activate the environment using:

```bash
workon rl2020
```

You should see then a parenthesis on your terminal as such:
```bash
(rl2020) [vulcan]s1873000:
```
Now clone the environment:
```bash
git clone https://github.com/semitable/uoe-rl2020
```

Navigate and install using:
```bash
cd uoe-rl2020
pip install -e .
```

## FYI Your Forks are PUBLIC!
Forks in github are public, and usually easy to find by clicking the number next to your forks.

If you want to mirror the repository you can do it this way: https://help.github.com/articles/duplicating-a-repository/ and don't forget to make it private afterwards.

You could then add the original remote and pull any updates (a bit less convenient but...)

using:
```bash
git remote add coursework https://github.com/semitable/uoe-rl2020.git
```
and pulling as:
```bash
git pull coursework master
```
or just clone and work locally.

## PyTest

The test folder contains a few tests that should all pass for full marks. They include checks for usage of correct variable names and the existance of files.
To run them, navigate to the main folder `uoe-rl2020`
and run:
```bash
pytest -v
```


