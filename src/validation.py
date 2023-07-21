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
from operator import add

# Sum of Absolute Differences (SAD) calculation for images
def SAD(self):
    
    def run_sad():
        
        # create dictionary for SAD calculation for each location repeat
        self.sad_correct = 0
        self.sad_mat = {}
        for n in self.locations:
            self.sad_mat[n] = np.array([])
        
        # create range for splitting out SAD matrices
        z = int(len(self.ims)/len(self.locations))
        zRange = [0]
        zAdd = [1]
        for m in range(1,len(self.locations)):
            zRange.append((z*m))
            zAdd.append(1)
        # loop through each test image to find the SAD
        for nEnum, n in enumerate(self.test_imgs):
            temp_vals = []
            # calculate SAD for each test image and store in dictionary
            for mEnum, m in enumerate(self.ims):
                temp_vals.append(1/(self.imWidth*self.imHeight)*np.sum(np.subtract(np.abs(m),np.abs(n))))

            if np.argmin(temp_vals) in zRange:
                self.sad_correct = self.sad_correct + 1
            
            zRange = list(map(add,zRange,zAdd))
            idx = [0,len(self.test_imgs)]
            for n, ndx in enumerate(self.locations):
                self.sad_mat[ndx] = np.append(self.sad_mat[ndx],np.array(temp_vals[idx[0]:idx[1]]))
                idx[0] = idx[0] + len(self.test_imgs)
                idx[1] = idx[1] + len(self.test_imgs)
        
            yield
        # determine if correct match
        print('SAD: number correct '+str(100*(self.sad_correct/len(self.test_imgs)))+'%')
        
    # run sum of absolute differences caluclation
    print('Calculating sum of absolute differences')
    with alive_bar(len(self.test_imgs)) as sbar:
       for i in run_sad():
          sbar()    

    # plot the similarity matrices
    sim_mat = np.zeros(len(self.test_imgs)*len(self.test_imgs))
    for n in self.sad_mat:
        sim_mat = sim_mat + self.sad_mat[n]
    sim_mat = sim_mat/len(self.locations)
    sim_mat = np.resize(sim_mat,(len(self.test_imgs),len(self.test_imgs)))
    plot_name = "Similarity Sum of Absolute Differences"
    fig = plt.figure()
    plt.matshow(sim_mat,fig, cmap=plt.cm.Greens)
    plt.colorbar(label="Pixel intensity")
    fig.suptitle(plot_name,fontsize = 12)
    plt.xlabel("Query",fontsize = 12)
    plt.ylabel("Database",fontsize = 12)
    plt.show()
    
    # run PR calculation for sum of absolute differences
    