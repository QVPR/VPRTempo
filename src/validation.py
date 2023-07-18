#MIT License

#Copyright (c) 2023 Adam Hines, Michael Milford, Tobias Fischer

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

'''
Imports
'''
import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar

# Sum of Absolute Differences (SAD) calculation for images
def SAD(self):
    
    print("Calculating Sum of Absolute Differences (SAD) for training and testing data")
    def run_sad():
        numcorrect = 0
        test_mat = np.array([])
        # loop through each test image to find the SAD
        for nEnum, n in enumerate(self.test_imgs):
            
            # loop through each training image and calculate SAD
            test = []
            for mEnum, m in enumerate(self.ims):
                test.append(1/(self.imWidth*self.imHeight)*np.sum(np.subtract(np.abs(m),np.abs(n))))
            test = np.array(test)
            
            # calculate the precision-recall curves for each 
            # find the minimum value
            min_arg = np.argmin(test)
            if min_arg > int(self.train_img/self.location_repeat):
                if self.train_ids[min_arg-(int(self.train_img/self.location_repeat))] == self.test_ids[nEnum]:
                    numcorrect = numcorrect+1
            else:
                if self.train_ids[min_arg] == self.test_ids[nEnum]:
                    numcorrect = numcorrect+1

            test_mat = np.concatenate((test_mat,test),axis=0)
        
        # create the similarity matrix
        self.sim_mat = np.copy(np.reshape(test_mat,(self.test_t,self.train_img)))
        
        # store the output
        self.SAD_correct = 100*(numcorrect/len(self.test_imgs)) 

            
    run_sad()
    print('Number of correct images with SAD: '+str(self.SAD_correct)+'%')
    
# Precision @ 100% recall calculation
def PR_calc():
    
    print("Calculating precision @ 100% recall")