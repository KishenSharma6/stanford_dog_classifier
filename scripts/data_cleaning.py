import os
import sys

def lowercase_directories(path):
    #renames subdirectories in path to all lowercase
    for file in os.listdir(path):
        os.rename(path + file, path + file.lower())

    print("subdirectories in " + (path) + " have been converted to lowercase")


if __name__ == "__main__":
    path = str(sys.argv[1])
    lowercase_directories(path)