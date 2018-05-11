# coding=utf-8
# modify for run on windows with python3.5
__author__ = 'Aleksandar Gyorev and modify by illool@163.com'
__email__ = 'a.gyorev@jacobs-university.de'

import os
import cv2
import sys
import shutil
import random
import numpy as np

"""
A Python class that implements the Eigenfaces algorithm
for face recognition, using eigenvalue decomposition and
principle component analysis.

We use the AT&T data set, with 60% of the images as train
and the rest 40% as a test set, including 85% of the energy.

Additionally, we use a small set of celebrity images to
find the best AT&T matches to them.

Example Call:
    $> python3.5 eigenfaces.py att_faces celebrity_faces

Algorithm Reference:
    http://docs.opencv.org/modules/contrib/doc/facerec/facerec_tutorial.html
"""


# *** COMMENTS ***
class Eigenfaces(object):
    faces_count = 40

    # directory path to the AT&T faces
    faces_dir = '.'

    # number of faces used for training
    train_faces_count = 6
    # number of faces used for testing
    test_faces_count = 4

    # training images count
    l = train_faces_count * faces_count
    # number of columns of the image
    m = 92
    # number of rows of the image
    n = 112
    # length of the column vector
    mn = m * n

    """
    Initializing the Eigenfaces model.
    """

    def __init__(self, _faces_dir='.', _energy=0.85):
        print('> Initializing started')

        self.faces_dir = _faces_dir
        self.energy = _energy
        # train image id's for every at&t face
        self.training_ids = []

        # each row of L represents one train image
        L = np.empty(shape=(self.mn, self.l), dtype='float64')

        cur_img = 0
        for face_id in range(1, self.faces_count + 1):

            # the id's of the 6 random training images
            training_ids = random.sample(range(1, 11), self.train_faces_count)
            # remembering the training id's for later
            self.training_ids.append(training_ids)

            for training_id in training_ids:
                path_to_img = os.path.join(self.faces_dir,
                                           's' + str(face_id), str(training_id) + '.pgm')          # relative path
                # print '> reading file: ' + path_to_img

                # read a grayscale image
                img = cv2.imread(path_to_img, 0)
                # flatten the 2d image into 1d
                img_col = np.array(img, dtype='float64').flatten()

                # set the cur_img-th column to the current training image
                L[:, cur_img] = img_col[:]
                cur_img += 1

        # get the mean of all images / over the rows of L
        self.mean_img_col = np.sum(L, axis=1) / self.l

        # subtract from all training images
        for j in range(0, self.l):
            L[:, j] -= self.mean_img_col[:]

        # instead of computing the covariance matrix as
        C = np.matrix(L.transpose()) * np.matrix(L)
        # L*L^T, we set C = L^T*L, and end up with way
        C /= self.l
        # smaller and computentionally inexpensive one
        # we also need to divide by the number of training
        # images

        # eigenvectors/values of the covariance matrix
        self.evalues, self.evectors = np.linalg.eig(C)
        # getting their correct order - decreasing
        sort_indices = self.evalues.argsort()[::-1]
        # puttin the evalues in that order
        self.evalues = self.evalues[sort_indices]
        # same for the evectors
        self.evectors = self.evectors[sort_indices]

        # include only the first k evectors/values so
        evalues_sum = sum(self.evalues[:])
        # that they include approx. 85% of the energy
        evalues_count = 0
        evalues_energy = 0.0
        for evalue in self.evalues:
            evalues_count += 1
            evalues_energy += evalue / evalues_sum

            if evalues_energy >= self.energy:
                break

        # reduce the number of eigenvectors/values to consider
        self.evalues = self.evalues[0:evalues_count]
        self.evectors = self.evectors[0:evalues_count]

        # change eigenvectors from rows to columns
        self.evectors = self.evectors.transpose()
        # left multiply to get the correct evectors
        self.evectors = L * self.evectors
        # find the norm of each eigenvector
        norms = np.linalg.norm(self.evectors, axis=0)
        # normalize all eigenvectors
        self.evectors = self.evectors / norms

        # computing the weights
        self.W = self.evectors.transpose() * L

        print('> Initializing ended')

    """
    Classify an image to one of the eigenfaces.
    """

    def classify(self, path_to_img):
        # read as a grayscale image
        img = cv2.imread(path_to_img, 0)
        img_col = np.array(img, dtype='float64').flatten(
        )                      # flatten the image
        # subract the mean column
        img_col -= self.mean_img_col
        # from row vector to col vector
        img_col = np.reshape(img_col, (self.mn, 1))

        # projecting the normalized probe onto the
        S = self.evectors.transpose() * img_col
        # Eigenspace, to find out the weights

        # finding the min ||W_j - S||
        diff = self.W - S
        norms = np.linalg.norm(diff, axis=0)

        # the id [0..240) of the minerror face to the sample
        closest_face_id = np.argmin(norms)
        # return the faceid (1..40)
        return (closest_face_id / self.train_faces_count) + 1

    """
    Evaluate the model using the 4 test faces left
    from every different face in the AT&T set.
    """

    def evaluate(self):
        print('> Evaluating AT&T faces started')
        # filename for writing the evaluating results in
        #results_file = os.path.join('results', 'att_results.txt')
        path = 'results' + "\\" + 'att_results.txt'
        pwd = os.getcwd()
        father_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + ".")
        print(father_path)
        fina_path = os.path.join(father_path, path)
        print(fina_path)
        # the actual file
        f = open(fina_path, 'w')

        # number of all AT&T test images/faces
        test_count = self.test_faces_count * self.faces_count
        test_correct = 0
        for face_id in range(1, self.faces_count + 1):
            for test_id in range(1, 11):
                # we skip the image if it is part of the training set
                if (test_id in self.training_ids[face_id - 1]) == False:
                    path_to_img = os.path.join(self.faces_dir,
                                               's' + str(face_id), str(test_id) + '.pgm')          # relative path

                    result_id = self.classify(path_to_img)
                    result = (result_id == face_id)

                    if result == True:
                        test_correct += 1
                        f.write('image: %s\nresult: correct\n\n' % path_to_img)
                    else:
                        f.write('image: %s\nresult: wrong, got %2d\n\n' %
                                (path_to_img, result_id))

        print('> Evaluating AT&T faces ended')
        self.accuracy = float(100. * test_correct / test_count)
        print('Correct: ' + str(self.accuracy) + '%')
        f.write('Correct: %.2f\n' % (self.accuracy))
        # closing the file
        f.close()

    """
    Evaluate the model for the small celebrity data set.
    Returning the top 5 matches within the AT&T set.
    Images should have the same size (92,112) and are
    located in the celebrity_dir folder.
    """

    def evaluate_celebrities(self, celebrity_dir='.'):
        print('> Evaluating celebrity matches started')
        # go through all the celebrity images in the folder
        for img_name in os.listdir(celebrity_dir):
            path_to_img = os.path.join(celebrity_dir, img_name)

            # read as a grayscale image
            img = cv2.imread(path_to_img, 0)
            img_col = np.array(img, dtype='float64').flatten(
            )                  # flatten the image
            # subract the mean column
            img_col -= self.mean_img_col
            # from row vector to col vector
            img_col = np.reshape(img_col, (self.mn, 1))

            # projecting the normalized probe onto the
            S = self.evectors.transpose() * img_col
            # Eigenspace, to find out the weights

            # finding the min ||W_j - S||
            diff = self.W - S
            norms = np.linalg.norm(diff, axis=0)
            # first five elements: indices of top 5 matches in AT&T set
            top5_ids = np.argpartition(norms, 5)[:5]

            # the image file name without extension
            name_noext = os.path.splitext(img_name)[0]
            # path to the respective results folder
            result_dir = os.path.join('results', name_noext)
            # make a results folder for the respective celebrity
            os.makedirs(result_dir)
            # the file with the similarity value and id's
            result_file = os.path.join(result_dir, 'results.txt')

            # open the results file for writing
            f = open(result_file, 'w')
            for top_id in top5_ids:
                # getting the face_id of one of the closest matches
                face_id = (top_id / self.train_faces_count) + 1
                # getting the exact subimage from the face
                subface_id = self.training_ids[face_id -
                                               1][top_id % self.train_faces_count]

                path_to_img = os.path.join(self.faces_dir,
                                           's' + str(face_id), str(subface_id) + '.pgm')           # relative path to the top5 face

                shutil.copyfile(path_to_img,                                    # copy the top face from source
                                os.path.join(result_dir, str(top_id) + '.pgm'))         # to destination

                # write the id and its score to the results file
                f.write('id: %3d, score: %.6f\n' % (top_id, norms[top_id]))

            # close the results file
            f.close()
        print('> Evaluating celebrity matches ended')


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('Usage: python3.5 eigenfaces.py '
              + '<att faces dir> [<celebrity faces dir>]')
        sys.exit(1)

    # create a folder where to store the results
    if not os.path.exists('results'):
        os.makedirs('results')
    else:
        # clear everything in the results folder
        shutil.rmtree('results')
        os.makedirs('results')

    # create the Eigenfaces object with the data dir
    efaces = Eigenfaces(str(sys.argv[1]))
    # evaluate our model
    efaces.evaluate()

    # if we have third argument (celebrity folder)
    if len(sys.argv) == 3:
        # find best matches for the celebrities
        efaces.evaluate_celebrities(str(sys.argv[2]))
