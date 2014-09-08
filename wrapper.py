#!/usr/bin/python -u
import sys
import subprocess

# gtp coords are letters *** except I ***, number (from lower left)
board_letters = "abcdefghjklmnopqrstuvwxyz";
# note that sgf uses two letters from upper left, and includes I
sgf_letters = "abcdefghijklmnopqrstuvwxyz";

class Wrapper(object):
    def __init__(self):
        self.size = 19
        self.moves = ""
        self.komi = 6.5
        self.old_moves = {}
        pass

    def boardsize(self, arg):
        self.size = int(arg)
        print "="

    def clear_board(self):
        self.moves = ""
        print "="
        pass

    def set_free_handicap(self, *args):
        for loc in args:
            self.moves += "B" + loc
        print "="

    def komi(self, arg):
        self.komi = float(arg)
        print "="

    def play(self, colorname, xy):
        if xy in ['pass', 'resign']:
            print "=", xy
            return
        c =  "B" if colorname.lower().startswith('b') else "W"
        X = sgf_letters[board_letters.index(xy[0].lower())]
        Y = sgf_letters[self.size - int(xy[1:])]
        self.moves += c + X + Y
        print "="

    def genmove(self, arg):
        color_to_play =  "B" if arg.lower().startswith('b') else "W"
        key = (self.size, self.moves, self.komi)
        if key not in self.old_moves:
            result = subprocess.check_output(["/data/Ray/cudago/board",
                                              str(self.size),
                                              str(self.komi),
                                              str(self.moves),
                                              color_to_play])
            self.old_moves[key] = result
        result = self.old_moves[key]
        sys.stderr.write("RECEIVED %s\n" % result)
        if result.strip() == "pass":
            print "= pass"
        elif result.strip() == "resign":
            print "= resign"
        else:
            x, y = result.strip().split(' ')
            tosend = "= %s%d" % (board_letters[int(x) - 1], self.size - int(y) + 1)
            sys.stderr.write("SENDING " + tosend  + "\n")
            print tosend

    def name(self):
        print "= CudaGo"

    def protocol_version(self):
        print "= 2"

    def version(self):
        print "= 0.5"

    def command_list(self):
        for s in ["boardsize",
                  "clear_board",
                  "genmove",
                  "list_commands",
                  "name",
                  "play",
                  "protocol_version",
                  "set_free_handicap",
                  "showboard",
                  "komi",
                  "version"]:
            print s

    def __call__(self, command_str):
        parts = command_str.split(" ")
        command = parts[0]
        args = parts[1:]
        if hasattr(self, command):
            getattr(self, command)(*args)
            print ""
        else:
            print "? Unknown command: %s" % command


if __name__ == "__main__":
    wrapper = Wrapper()

    sys.stderr.write("STARTING WRAPPER\n");
    while True:
        l = sys.stdin.readline()
        sys.stderr.write("received " + l)
        wrapper(l.strip())
