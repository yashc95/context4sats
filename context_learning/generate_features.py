import os
import collections

class featureExtractor:
    def __init__(self, folder):
        # Folder is the folder for the yolo_labels of the
        self.folder = folder
        self.files = []
        self.extractFiles()
        self.num_classes = 4
    def extractFiles(self):
        directory = os.fsencode(self.folder + '/yolo_labels/')
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                self.files.append(filename[0:-4])

    def subImageCounts(self,file, useDists = False):
        """
        Generates features using surrounding object counts for each object in the provided image
        :param file: The number of the image you want to look at (everything in the name besides the extensions)
                useDists: whether to include the average distance to each class in the feature list
        :return: [(feats, label)]: a list of feature- label pairs, where the features are counts for each class in the image
        excluding one particular object (so there is a pair for each object in the image) plus (optionally)
        the average distance to each class occurrence. The label is the true class of the object
        So feature vector with useDists enabled is [class 0 count, class 1 count,...class 3 count, avg of class 0 dist, ..., avg of class 3 dist]
        """
        # First gather the rows in the yolo_labels file for the given image:
        lines = []
        with open(self.folder + '/yolo_labels/' + file + '.txt') as f:
            for line in f:
                lines.append(line)
        out = []
        for i in range(len(lines)):
            c = collections.Counter()
            words = lines[i].split()
            label = int(words[0])
            if useDists:
                d = {'0': 0, '1': 0, '2': 0, '3':0}
                x = float(words[1])
                y = float(words[2])
            for j in range(len(lines)):
                if i != j:
                    rowwords = lines[j].split()
                    rowclass = rowwords[0]
                    rowx = float(rowwords[1])
                    rowy = float(rowwords[2])
                    c.update(rowclass)
                    if useDists:
                        dist = ((rowy - y) ** 2 + (rowx - x) ** 2) ** (0.5)
                        d[rowclass] += dist
            feats = []
            # Add class counts
            for keynum in range(self.num_classes):
                if str(keynum) in c.keys():
                    feats.append(c[str(keynum)])
                else:
                    feats.append(0)
            if useDists:
                for keynum in range(self.num_classes):
                    if str(keynum) in c.keys():
                        feats.append(1/(d[str(keynum)]/c[str(keynum)]))
                    else:
                        feats.append(d[str(keynum)])
            out.append((feats,label))
        return out
# Add directionality to macro image objects
# Encode color
# Usage Example
f = featureExtractor('./split_test_clean_balanced')
feat_pairs = f.subImageCounts('P0128__1__633___0',True)
"""
