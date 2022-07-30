// SPDX-License-Identifier: MIT
pragma solidity ^0.8.11;

// We save the following things in the smart contract
contract decentraldl{
    uint sumOfNum = 0;
    int [1024][1] weights; // The load state dictionary has inverted shape
    int [1] bias;
    int totalNumberOfImages = 0;
    int [1024] normalizedWeights;
    int [1024] beta;

    int [1024][1] addWeights;
    int [1] addBias;
    int [1024] addNormalizedWeights;
    int [1024] addBeta;
    
   // function to get initial weights from hospital and store them
   function storeInitialWeights(int i,int j,int val) public{
       uint m = uint(i);
       uint n = uint(j);
       weights[m][n] = val;

   }
   
   function returnInitialWeights(uint i) public view returns(int[1024] memory){
       return weights[i];
   }
  
   function storeNormalizedWeights(int i, int val) public{
       uint p = uint(i);
       normalizedWeights[p] = val;
   }
   function returnNormalizedWeights() public view returns (int[1024] memory){
       return normalizedWeights;
   }
    int total_samples = 0;
   // functions to average out the weights
   // function to add total number of samples - also after every epoch - these need to be updated to 0
   function addSamples(int val) public{
       total_samples = total_samples + val;
   }
   function returnSamples() public view returns(int){
       return total_samples;
   }
   // function to divide weights
   
   function restartNsamples() public {
       total_samples = 0;
   }

    function storeBias(int i, int val) public{
       uint p = uint(i);
       bias[p] = val;
   }
   function storeBeta(int i, int val) public{
       uint p = uint(i);
       beta[p] = val;
   }

   function returnBias() public view returns (int[1] memory){
       return bias;
   }
   function returnBeta() public view returns (int[1024] memory){
       return beta;
   }
  
    /// functions with new and proper logic

    function resetWeights(int i, int j) public{
        uint k = uint(i);
        uint m = uint(j);
                addWeights[k][m] =0;
            
        
    }
    function resetBias(int m) public{
        uint i = uint(m);
            addBias[i] = 0;
        
    }
    function resetNormalizedweights(uint n) public{
       uint j = uint(n);
            addNormalizedWeights[j] =0;

    

    }
    function resetBeta(int p) public{
       uint j = uint(p);
  
            addBeta[j] =0;
        
        
    }
    function updateAddWeights(int i, int j, int val) public{
        uint m = uint(i);
        uint n = uint(j);
        addWeights[m][n] = addWeights[m][n] + val;

    }
    function updateAddNormalizedWeights(int i, int val) public{
        uint n = uint(i);
        addNormalizedWeights[n] = addNormalizedWeights[n] + val;
    }
    function updateAddBeta(int i, int val) public{
             uint r = uint (i);
       addBeta[r] = addBeta[r]+val;
    }
    function updateAddBias(int i, int val) public{
             uint r = uint (i);
       addBias[r] = addBias[r]+val;
    }

    /// Average out the weights
    function averageBias(int i) public{
       uint p = uint(i);
       //bias[p] = ((bias[p]*total_samples)+addBias[p])/total_samples;
       bias[p] = addBias[p]/total_samples;
   }
   function averageBeta(int i) public{
       uint p = uint(i);
       //beta[p] = ((beta[p]*total_samples)+addBeta[p])/total_samples;
       beta[p] = addBeta[p]/total_samples;
   }

   function averageWeights(int i,int j) public {
       uint x = uint(i);
       uint y = uint(j);
       //weights[x][y] = weights[x][y]/total_samples;
       weights[x][y] = addWeights[x][y]/total_samples;
       //weights[x][y] = ((weights[x][y]*total_samples)+addWeights[x][y])/total_samples;
   }

   // function to divide normalized weights
     function averageNormalizedWeights(int i) public{
       uint p = uint(i);
       //normalizedWeights[p] = normalizedWeights[p]/total_samples;
       normalizedWeights[p] = addNormalizedWeights[p]/total_samples;
       //normalizedWeights[p] = ((normalizedWeights[p]*total_samples)+addNormalizedWeights[p])/total_samples;
   }
   int [2048][1000] resnetweights; // The load state dictionary has inverted shape
    function storeresNetWeight(int i, int j, int val)public{

        uint m = uint(i);
        uint n = uint(j);
        resnetweights[m][n] =  val;

    

    }

    
 
}



