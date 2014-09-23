#include "ScoreMatrix.h"
#include "Score.h"
#include "bCNF.h"

using namespace std;

#define DUMMY -1

class bCNF_Model;

class SEQUENCE
{
public:
	SEQUENCE(int len, bCNF_Model* pModel);
	~SEQUENCE();

	bCNF_Model* m_pModel;

	int length_seq;
    int q3;
    
	int* obs_label;
	Score *_features;
	Score **obs_feature;

	Score Partition;
	int *predicted_label;
	ScoreMatrix *forward;
	ScoreMatrix *backward;
  double viterbiprob;
  double mapprob;
    //    int *predicted_label;

    vector<Score*> allresults;
    //every element of the vector allresults stores the classification result of one model.

    Score* predicted_allstates;
    
	ScoreMatrix gates;
	ScoreMatrix arrVi;
	void ComputeGates();
	void ComputeVi();
  void modifilabels(int);
  void modififeatures(int);
  
	void ComputeViterbi();
	void ComputeForward();
	void ComputeBackward();
	void CalcPartition();

	void ComputePartition();
	Score ComputeScore(int leftState, int currState, int pos);

	void makeFeatures();
	Score* getFeatures(int pos);

	int GetObsState(int pos);
	int GetObsLabel(int pos);

	void ComputeGradient(bool bCalculateGate);
	void MAP();	
	void MAP1();	
	void MAP2();	
	void onebest();
    
	void ComputeTestAccuracy();
    void ComputeTestAccuracyL8();
    
	Score Obj();
	Score Obj_p();
    void Voting();
    void ComputeConfusionMatrix();

};	

class _LBFGS: public Minimizer
{
public:
	_LBFGS(bCNF_Model* pModel): Minimizer(false) {m_pModel = pModel;};

	bCNF_Model* m_pModel;

	void ComputeGradient(vector<double> &g, const vector<double> &x, bool bCalculateGate);
	double ComputeFunction(const vector<double> &x);
	void Report(const vector<double> &theta, int iteration, double objective, double step_length);
	void Report(const string &s);
};

class bCNF_Model
{
public:
	int num_states;
	int num_label;
	int num_data;
	int num_tst;
	int num_gates;
    int num_ps_gates;
    int num_pi_gates;
    int dim_ps;

    int dim_pi;
    int phase;
  int featmask;
	int dim_one_pos;
	int dim_features;
	int window_size;
	int num_params;
	int totalPos;
    int LocalWeights0;
    int PsInterWeights0;
    int PiInterWeights0;
    int l83[8];
  int bien;
  double totalmapprob;
  double totalviterbiprob;
    int nModel; // 1st_order or 2nd_order
    enum{first_order=1, second_order=2};
  char * jobid;


    map<string, string> params;
	int totalCorrect;
int dim_one_pos_sparse;
 int dim_dense;
	string model_file;
    string trndata,tstdata,workdir;
    string retrndata,retstdata;
    
	double apw;
	int ct;
    int model_num;
    vector<string> models;
    vector<string> testfiles;
    vector<string> trainfiles;
	vector<SEQUENCE*> trainData,testData;
    
	double* grad;
	double* weights;
	double* grad_sum;
	double* reg;
	~bCNF_Model(){
	}
	// get the transition prob, num_states( for DUMMY ) + num_states*num_states
	//	inline Score GetWeightT(int leftState, int currState){return weights[(1 + leftState)*num_states + currState];}

	// get the label-based weight, num_states*num_gates
	//inline Score GetWeightL(int currState, int gate){ return weights[num_states + num_states*num_states + currState*num_gates + gate];}

	// get the Gate weight, num_gates*dim_features
	//inline Score GetWeightG(int gate, int dim){ return weights[num_states*(num_states+num_gates + 1) + dim_features*gate + dim];}
	void SetSeed();
    void Ensemble_start();//by wzy
    void Double_train();//by wzy, train the model using the output of several models.
    void ReloadData(string testfile,string trainfile);
    
	void SetParameters(int, char**);
	void Initialize(int, char**);
	void LoadData();
    void LoadRetrainData();
    
	Score Gate(Score sum){ return (Score)1.0/(1.0+exp(-(double)sum)); }
    
    //[TP]revised in tp version, weighted sum part of all the features only this gate cares about
	Score GetGateOutput(int gate, Score* features){
		Score output = 0;
		//dim_features
        //[TP]
        if(gate<num_ps_gates)
        {
          int weightGStart = PsInterWeights0 + (window_size*(dim_ps + 1)+1)*gate;
          for(int w=0;w<window_size;w++)
          {
            int pos;
            
            for(int j=0;j<dim_ps;j++)
            {
			//output+=GetWeightG(gate,j)*features[j];
              pos = w * (dim_dense+1) + j;
              output += weights[weightGStart++] * features[pos];
            }
            output += weights[weightGStart++] * features[w*(dim_dense+1)+dim_dense];//local bias
          }
          output += weights[weightGStart++] * 1; //global bias
          if(isnan(Gate(output)))throw -1;
          
          return Gate(output);
        }
        //        if(gate>=num_gates)
        //{
        int weightGStart = PiInterWeights0 + (gate - num_ps_gates) * window_size *2;
        for(int w=0;w<window_size;w++)
        {
			//output+=GetWeightG(gate,j)*features[j];
          int pos = w * (dim_dense +1) + gate - num_ps_gates + dim_ps;// gate - num_ps_gates < dim_pi;
          output += weights[weightGStart++] * features[pos];
          int posb = w * (dim_dense + 1) + dim_dense;
          output += weights[weightGStart++] * features[posb];//local bias
          if(isnan(output))throw -1;
          
        }
if(isnan(Gate(output)))throw -1;
        return Gate(output);
          //}
        
	}
};
