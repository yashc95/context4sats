import os
import collections

class featureExtractor:
    def __init__(self, folder):
        self.folder = folder
        self.files = []
        self.extractFiles()

    def extractFiles(self):
        directory = os.fsencode(self.folder + 'labels/')
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                self.files.append(filename[0:-4])
