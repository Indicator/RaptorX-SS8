#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#ifdef _MPI
#include <mpi.h>
#endif
#include <sys/stat.h>
#include "bCNF_mpi_tp.h"
SEQUENCE::SEQUENCE(int length_seq, bCNF_Model* pModel)
{
	m_pModel = pModel;
	this->length_seq = length_seq;
	forward = new ScoreMatrix(m_pModel->num_states,length_seq);
	backward = new ScoreMatrix(m_pModel->num_states,length_seq);
	obs_label = new int[length_seq];
	int df = m_pModel->dim_features*length_seq;
	_features = new Score[df];
	q3=0;
	predicted_label = new int[length_seq];
	predicted_allstates=new Score[length_seq*m_pModel->num_label];
	obs_feature = new Score*[length_seq];
	for(int i=0; i<length_seq; i++)
		obs_feature[i] = new Score[m_pModel->dim_one_pos+1];
}
SEQUENCE::~SEQUENCE()
{
	delete forward;
	delete backward;
	delete obs_label;
	delete _features;
	for(int i=0; i<length_seq; i++)
		delete obs_feature[i];
	delete obs_feature;
}
Score SEQUENCE::Obj()  // observed label : log-likelihood`
{
	ComputeGates();
	ComputeVi();
	ComputeForward();
	CalcPartition();
	
	Score obj = -Partition;
//	if(proc_id==0)cerr<<"obj() partition "<<obj<<endl;
	for(int t=0; t<length_seq; t++) {
		int leftState = GetObsState(t-1);
		int currState = GetObsState(t);
		if(leftState % m_pModel->num_label != currState / m_pModel->num_label -1 ) {
			cerr<<"getobsstate error";
			cerr<< leftState<<" "<<currState<<"\n";
			continue;
		}
		Score temp=ComputeScore(leftState,currState,t);
		obj+=temp;
	}
//	if(proc_id==0)cerr<<"obj() "<<obj<<endl;
	return obj;
	
}
void SEQUENCE::ComputeViterbi()
{
	ComputeForward();
	ComputeBackward();
	CalcPartition();
	// Viterbi Matrix
	ScoreMatrix best(m_pModel->num_states,length_seq);
	best.Fill((Score)LogScore_ZERO);
	// TraceBack Matrix
	ScoreMatrix traceback(m_pModel->num_states,length_seq);
	traceback.Fill(DUMMY);
	// compute the scores for the first position
	for(int i=0; i<m_pModel->num_label; i++) {
		best(i,0)=ComputeScore(DUMMY,i,0);
	}
	// Compute the Viterbi Matrix
	for(int t=1; t<length_seq; t++) {
		for(int currState=m_pModel->num_label; currState<m_pModel->num_states; currState++) {
			for(int leftState=0; leftState<m_pModel->num_states; leftState++) {
				if(leftState % m_pModel->num_label != currState / m_pModel->num_label -1 ) {
					continue; // state= (a+1) * num_label+b, 
				}
				if((t==1 && leftState >= m_pModel->num_label)||
				   (t>1 && leftState < m_pModel->num_label)) {
					continue;
				}
				Score new_score;
				new_score = ComputeScore(leftState,currState,t) + best(leftState,t-1);
        if(isnan(new_score))throw -1;
        //        cerr<<" "<<new_score;
				if(new_score > best(currState,t)) {
					best(currState,t) = new_score;
          //cerr<<endl;
					traceback(currState,t) = leftState;
				}
        //cerr<<endl;
        
			}
		}
	}
	Score max_s = LogScore_ZERO;
	int last_state = 0;
	//Find the best last state.
	for(int i=m_pModel->num_label; i<m_pModel->num_states; i++){
    //cerr<<" "<<best(i,length_seq-1);
		if(best(i,length_seq-1)>max_s)
			max_s = best(i,length_seq-1), last_state = i;
  }
  
	viterbiprob=max_s-Partition;
  //if(last_state<m_pModel->num_label)
  //  last_state+=m_pModel->num_label; // the last state should not be one of the beginning state.

	//TraceBack
	for(int t=length_seq-1; t>=0; t--) {
		predicted_label[t]=last_state % m_pModel->num_label;
		int leftState=(int)traceback(last_state,t);
		last_state=leftState; // % m_pModel->num_label;
		predicted_label[t]=predicted_label[t]/2;
	}
}
void SEQUENCE::MAP1()  // Posterior Decoding (Marginal Probability Decoder)
{
	ComputeForward();
	ComputeBackward();
	CalcPartition();
	mapprob=LogScore_ZERO;
	for(int t=0; t<length_seq; t++) {
		int idx = 0;
		Score maxS = LogScore_ZERO;
		for(int i=0; i<m_pModel->num_label; i++)
			predicted_allstates[t*m_pModel->num_label+i]=LogScore_ZERO;
		int pos = t*m_pModel->num_label;
		//        predicted_allstates[pos] -= Partition;//Store per AA probability
		for(int i=0; i<m_pModel->num_states; i++) {
			if ((t==0 && i>=m_pModel->num_label) ||(t>0 && i<m_pModel->num_label) )continue;
			//          int label=num_states % m_pModel->num_label;
			Score s = (*backward)(i,t) + (*forward)(i,t) - Partition;
			int pos = t*m_pModel->num_label+ i % m_pModel->num_label;
			LogScore_PLUS_EQUALS(predicted_allstates[pos], s);
//       if(predicted_allstates[pos] > maxS){
//         maxS = predicted_allstates[pos];
//         idx = i % m_pModel->num_label;
//       }
		}
		for(int i=0; i<m_pModel->num_label/2; i++) { //num_label has been doubled.
			int pos = t*m_pModel->num_label + i*2;
			LogScore_PLUS_EQUALS(predicted_allstates[pos], predicted_allstates[pos+1]);
			if(predicted_allstates[pos] > maxS) {
				maxS = predicted_allstates[pos];
				idx = i ;
			}
		}
		mapprob+=predicted_allstates[t*m_pModel->num_label+obs_label[t]];
		predicted_label[t]=idx;
	}
}
void SEQUENCE::ComputeForward()
{
	forward->Fill(LogScore_ZERO);
	for(int i=0; i<m_pModel->num_label; i++) {
		(*forward)(i,0)=ComputeScore(DUMMY,i,0);
	}
	for(int t=1; t<length_seq; t++) {
		//from t=1,leftstate cannot be DUMMY,DUMMY
		for(int currState=m_pModel->num_label; currState<m_pModel->num_states; currState++) {
			//Sum of combination prob from different left state(2-label) to currlabel.
			for(int leftStateL=-1; leftStateL<m_pModel->num_label; leftStateL++) {
				int leftState= (leftStateL+1) *m_pModel->num_label + currState / m_pModel->num_label -1;
				//if(leftState % m_pModel->num_label != currState / m_pModel->num_label -1 )
				// {
				//    continue;
				//  }
				if( (t==1 && leftState >= m_pModel->num_label) || (t>1 && leftState < m_pModel->num_label) ) {
					continue;
				}
				Score new_score = ComputeScore(leftState,currState,t) + (*forward)(leftState,t-1) ;
				LogScore_PLUS_EQUALS((*forward)(currState,t),new_score);
			}
		}
	}
//	if(proc_id==0){
//		cerr<<endl;
//		cerr<<"ComputeForward forward, ";
//		for(int k=m_pModel->num_label; k<m_pModel->num_states; k++){
//			cerr<<(*forward)(k,length_seq-1)<<" ";
//		}
//		cerr<<endl;
//	}
}
void SEQUENCE::ComputeBackward()
{
	//Compute p(Xn+1,..XN|Zn )
	backward->Fill(LogScore_ZERO);
	for(int i=0; i<m_pModel->num_states; i++) {
		(*backward)(i,length_seq-1)=0;
	}
	for(int t=length_seq-2; t>=0; t--) {
		for(int currState=0; currState<m_pModel->num_states; currState++) {
			//        for(int rightStateR=m_pModel->num_label;rightStateR<m_pModel->num_states;rightStateR++){
			for(int rightStateR=0; rightStateR<m_pModel->num_label; rightStateR++) {
				int rightState= (currState % m_pModel->num_label + 1) * m_pModel->num_label + rightStateR;
				/*
				  if(rightState / m_pModel->num_label -1 != currState % m_pModel->num_label )
				  {
				  continue;
				  }*/
				if((t==0 && currState >=m_pModel->num_label )||(t>0 && currState < m_pModel->num_label ) ) {
					continue;
				}
				Score new_score = ComputeScore(currState,rightState,t+1)+ (*backward)(rightState,t+1);
				LogScore_PLUS_EQUALS((*backward)(currState,t),new_score );
			}
		}
	}
}
void  SEQUENCE::modififeatures(int featmask)
{
	for(int j=0; j<length_seq; j++) {
		for(int k=0; k<m_pModel->dim_one_pos; k++) {
			//        trn_in >> seq->obs_feature[j][k];
			if(featmask==1) {
				//only use PSSM
				if(k<1||k>20)obs_feature[j][k]=0;
			}
			if(featmask==2) {
				//use pssm and neff
				if(k>20)obs_feature[j][k]=0;
			}
			if(featmask==3) {
				//use pssm and jufo
				if(k<1||k>20+7)obs_feature[j][k]=0;
			}
			if(featmask==4) {
				//use pssm and ss ends
				if(k<1||(k>20 && k<= 27)|| k> 38)obs_feature[j][k]=0;
			}
			if(featmask==5) {
				//use pssm and kihara CCPC_KOLA
				if(k<1||(k>20 && k<=38)||k>78)obs_feature[j][k]=0;
			}
			if(featmask==6) {
				//use pssm and identity
				if(k<1||(k>20 && k<=78))obs_feature[j][k]=0;
			}
			if(featmask==200) {
				// use all matrices but pssm
				if(k<21)obs_feature[j][k]=0;
			}
			if(featmask==22) {
				//use  neff
				if(k>0)obs_feature[j][k]=0;
			}
			if(featmask==23) {
				//use 5~ jufo
				if(k<21||k>20+7)obs_feature[j][k]=0;
			}
			if(featmask==24) {
				//use  ss ends
				if(k<21||(k>20 && k<= 27)|| k> 38)obs_feature[j][k]=0;
			}
			if(featmask==25) {
				//use  kihara CCPC
				if(k<21||(k>20 && k<=38)||k>58)obs_feature[j][k]=0;
			}
			if(featmask==251) {
				//use  kihara KOLA
				if(k<21||(k>20 && k<=58)||k>78)obs_feature[j][k]=0;
			}
			if(featmask==26) {
				//use  identity
				if(k<21||(k>20 && k<=78))obs_feature[j][k]=0;
			}
			if(featmask==13) {
				//use pssm and jufo + neff
				if(k>20+7)obs_feature[j][k]=0;
			}
			if(featmask==14) {
				//use pssm and ss ends + neff
				if((k>20 && k<= 27)|| k> 38)obs_feature[j][k]=0;
			}
			if(featmask==15) {
				//use pssm and kihara CCPC_KOLA + neff
				if((k>20 && k<=38)||k>78)obs_feature[j][k]=0;
			}
		}
		//normalization
		vector<double> means;
		vector<double> stds;
		
		if(m_pModel->params["NORM"]!=""){
			ifstream ifnorm(m_pModel->params["NORM"].c_str());
			for(int hh=0;hh<=m_pModel->dim_one_pos;hh++){
				double x;
				ifnorm>>x;
				means.push_back(x);
			}
			for(int hh=0;hh<=m_pModel->dim_one_pos;hh++){
				double x;
				ifnorm>>x;
				stds.push_back(x);
			}
			ifnorm.close();
		}else{
      means.resize(m_pModel->dim_one_pos+1,0);
      stds.resize(m_pModel->dim_one_pos+1,1);
    }
    
		for(int k=0; k<m_pModel->dim_one_pos; k++) {
			obs_feature[j][k]=(obs_feature[j][k]-means[k+1])/stds[k+1];
		}
		//seq->obs_feature[j][dim_one_pos] = 1;
	}
}
void  SEQUENCE::modifilabels(int bien)
{
	vector<int> bielabel(length_seq);
	for(int j=0; j<length_seq; j++) {
		bool flag=false;
		for(int k=1; k<=bien; k++) { //check if on begining of a segment of SS
			if(j+k<length_seq && obs_label[j]!=obs_label[j+k]) {
				flag=true;
				break;
			}
		}
		if(flag) {
			bielabel[j] = obs_label[j]*2 + 0;
		}
		flag=false;
		for(int k=1; k<=bien; k++) { //check in near end of a segment of SS
			if(j-k>=0 && obs_label[j]!=obs_label[j-k]) {
				flag=true;
				break;
			}
		}
		if(flag) { //if neither beginning nor end
			bielabel[j] = obs_label[j]*2 + 0;
		}
		bielabel[j] = obs_label[j]*2 + 1;
	}
	for(int j=0; j<length_seq; j++) {
		obs_label[j]=bielabel[j];
	}
}
void SEQUENCE::ComputeGates()
{
	gates.resize(m_pModel->num_gates, length_seq);
	for (int pos=0; pos<length_seq; pos++) {
		Score* features=getFeatures(pos);
		for (int k=0; k<m_pModel->num_gates; k++) {
			gates(k,pos) = m_pModel->GetGateOutput(k,features);
      if(isnan(gates(k,pos)))throw -1;
      
		}
	}
}
//This method
void SEQUENCE::ComputeVi()
{
	arrVi.resize(m_pModel->num_states, length_seq);
	int num_states = m_pModel->num_states;
	int num_gates = m_pModel->num_gates;
	for (int pos=0; pos<length_seq; pos++)
		for (int currState=0; currState<m_pModel->num_states; currState++) {
			if((pos==0 && currState>=m_pModel->num_label) ||
			   (pos>0 && currState<m_pModel->num_label)) {
				continue;
			}
			int weightLStart = m_pModel->LocalWeights0 + currState*num_gates;
			Score score = 0;
			//Mutiply neurons output and weights
			for(int k=0; k<m_pModel->num_gates; k++) {
				Score output = gates(k,pos);
        if(isnan(output))throw -1;
        if(isnan( m_pModel->weights[weightLStart++]))throw -1;
        
				score += m_pModel->weights[weightLStart++]*output;
			}
			arrVi(currState, pos) = score;
		}
}
Score SEQUENCE::ComputeScore(int leftState, int currState, int pos) //@@@
{
	//For 2-order model, leftstate is 2-label, currLabel is 1-label
	//Compute 2-label currState rom the leftState and currLabel
	int leftLabel;
	if(leftState!=DUMMY) {
		leftLabel=leftState % m_pModel->num_label ;
	}
	//if(currLabel >= m_pModel->num_label )  cout << "currstate:" << currState<< endl;
	int currLabel=currState % m_pModel->num_label;
	if((leftState!=DUMMY) && ((leftState %  m_pModel->num_label) != (currState /m_pModel->num_label -1))) {
    //		cerr<<"Check computescore error:"<<leftState %  m_pModel->num_label<<","<<currState /m_pModel->num_label -1<<"\n";
	}
	bool b1=(pos==0 && leftState!=DUMMY);
	bool b2=(pos==0 && currState >= m_pModel->num_label);
	bool b3=(pos>0 && currState< m_pModel->num_label );
	bool b4=(pos==1 && leftState==DUMMY );
	bool b5=(pos==1 && leftState>= m_pModel->num_label);
	bool b6=(pos>1 && leftState < m_pModel->num_label);
	if(b1||b2||b3||b4||b5||b6) {
		cerr<<"CSE";
	}
	Score score = m_pModel->weights[(1 + leftState)*m_pModel->num_label + currLabel]+arrVi(currState, pos);
	return score;
}
void SEQUENCE::Voting()
{
	int num_states=m_pModel->num_label;
	for(int i=0; i<length_seq; i++) {
		Score *avescore=new Score[num_states];
		for(int j=0; j<num_states; j++) {
			avescore[j]= LogScore_ZERO;
		}
		int kk=0;
		Score max=LogScore_ZERO;
		//count frequence of every state.
		for(vector<Score *>::iterator kiter=allresults.begin(); kiter!=allresults.end(); kiter++) {
			Score p=LogScore_ZERO;
			for(int l=0; l<num_states; l++) {
				LogScore_PLUS_EQUALS(p, (*kiter)[i*num_states+l]);
			}
			for(int l=0; l<num_states; l++) {
				(*kiter)[i*num_states+l]-=p;
				LogScore_PLUS_EQUALS(avescore[l],(*kiter)[i*num_states+l]);
			}
		}
		//    int idx=0,max=0;
		int idx;
		//    MPI_Barrier(MPI_COMM_WORLD);
		//MPI_Reduce(n,n, num_states, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		for(int j=0; j<num_states; j++) {
			if(avescore[j]>=max) {
				max=avescore[j];
				idx=j;
			}
		}
		predicted_label[i]=idx;
	}
}
void SEQUENCE::makeFeatures()
{
	//build features for local windows
	//deal with dense features and sparse features separately, said zy.
	for(int t=0; t<length_seq; t++) {
		int pivot = t*m_pModel->dim_features;
		for(int i=0; i<m_pModel->window_size; i++) {
			int offset = t+i-m_pModel->window_size/2;
			if(offset <0 || offset >=length_seq) {
				for(int j=0; j< m_pModel->dim_dense+1; j++)
					_features[pivot] = 0, pivot++;
				//          for(int j=dim_dense;j<dim_one_pos;j++)
				// _features[pivot]=0,pivot++;
			} else {
				for(int j=0; j<m_pModel->dim_dense+1; j++) {
					_features[pivot] = obs_feature[offset][j], pivot++;
				}
			}
		}
		//This is for sparse features, no need to consider it if not applied.
		for(int j= m_pModel->dim_dense+1; j< m_pModel->dim_one_pos; j++) {
			//        for(int i=0; i< window_size/3; i++)
			//{
			int offset = t;
			if(offset >=0 && offset <length_seq)
				_features[pivot]+=obs_feature[offset][j];
			//}
			pivot++;
		}
		//[TP removed]_features[pivot] = 1;
		_features[pivot] = 1;
	}
}
void SEQUENCE::CalcPartition()
{
	Partition = (Score)LogScore_ZERO;
	Score FP = (Score)LogScore_ZERO;
	for(int k=m_pModel->num_label; k<m_pModel->num_states; k++){
		LogScore_PLUS_EQUALS(Partition, (*forward)(k,length_seq-1));
		
	}
//	if(proc_id==0){
//		cerr<<"calc partition forward, ";
//		for(int k=m_pModel->num_label; k<m_pModel->num_states; k++){
//			cerr<<(*forward)(k,length_seq-1)<<" ";
//		}
//		cerr<<endl;
//	}
	//Score BP = (Score)LogScore_ZERO;
	//for(int k=0;k<m_pModel->num_states;k++) LogScore_PLUS_EQUALS(BP, (*backward)(k,0) + (*forward)(k,0));
}
Score* SEQUENCE::getFeatures(int pos)
{
	int offset;
	offset = pos* (m_pModel->dim_features);  //[TP]
	return _features+offset;
}
int SEQUENCE::GetObsState(int pos)
{
	if(pos<0 || pos>=length_seq) return DUMMY;
	if (1 || m_pModel->nModel==m_pModel->second_order) {
		if (pos>0)
			return (obs_label[pos-1]+1)*m_pModel->num_label + obs_label[pos];
	}
	return obs_label[pos];
}
int SEQUENCE::GetObsLabel(int pos)
{
	if(pos<0 || pos>=length_seq) return DUMMY;
	return obs_label[pos];
}
void SEQUENCE::ComputeGradient(bool bCalculateGate)
{
	int num_states = m_pModel->num_states;
	int num_label = m_pModel->num_label;
	int num_gates = m_pModel->num_gates;
	int LocalWeights0=m_pModel->LocalWeights0;
	int PsInterWeights0=m_pModel->PsInterWeights0;
	int PiInterWeights0=m_pModel->PiInterWeights0;
	int gstart = m_pModel->LocalWeights0; //num_states*(num_states+1);
	int fstart = m_pModel->PsInterWeights0; //Intergstart + num_states*num_gates;
	double* prob_weight_sum=new double[num_gates];
	int num_ps_gates = m_pModel->num_ps_gates;
	int dim_ps = m_pModel->dim_ps;
	int dim_pi = m_pModel->dim_pi;
	int dim_dense = m_pModel->dim_dense;
	int dim_features = m_pModel->dim_features;
	int window_size = m_pModel->window_size;
	// Compute backward forward and partition function;
	if (bCalculateGate) {
		ComputeGates();
		ComputeVi();
	}
	ComputeForward();
	ComputeBackward();
	CalcPartition();
	for(int t=0; t<length_seq; t++) {
		int leftState = GetObsState(t-1);
		int currLabel = GetObsLabel(t);
		int currState = GetObsState(t);
		Score* features=getFeatures(t);
		// Trans Weight, num_states maybe two label combo(2-order), one label(DUMMY. label ) and DUMMY. So, num_states+1
		//    cerr<<(1 + leftState)*num_label + currLabel<<" ";
		m_pModel->grad[(1 + leftState)*num_label + currLabel] += 1;
		for(int i=0; i<num_gates; i++) {
			//Score out = m_pModel->GetGateOutput(i,features); // Need to setup a cache for the gate output
			Score out = gates(i,t);
			//out_buf(i,t) = out;
			// label-based gates weights
			m_pModel->grad[ LocalWeights0 + currState*num_gates + i] += out;
			// Gate weights
			double weight_out = m_pModel->weights[LocalWeights0 + currState*num_gates + i]*(1.-out)*out;
			//[TP]
			if(i<num_ps_gates) {
				int gradStart = PsInterWeights0 + i*((dim_ps+1)*window_size+1);
				int pos;
				for(int w=0; w<window_size; w++) {
					for(int j=0; j<dim_ps; j++) {
						//inner parameters
						//m_pModel->grad[gradStart + j] += m_pModel->GetWeightL(currState,i)*(1.-out)*out*features[j];
						pos = w * (dim_dense+1) + j;
						m_pModel->grad[gradStart++] += weight_out*features[pos];
					}
					m_pModel->grad[gradStart++] += weight_out*features[w*(dim_dense+1)+dim_dense];
				}
				m_pModel->grad[gradStart++] += weight_out*1;
			}
			if(i>=num_ps_gates) {
				int gradGStart = PiInterWeights0 + (i - num_ps_gates) * window_size *2;
				for(int w=0; w<window_size; w++) {
					//output+=GetWeightG(gate,j)*features[j];
					int pos = w * (dim_dense+1) + i - num_ps_gates + dim_ps;// gate - num_gates < dim_pi;
					m_pModel->grad[gradGStart++] += weight_out*features[pos];
					int posb = w * (dim_dense + 1) + dim_dense; //local bias
					m_pModel->grad[gradGStart++] += weight_out*features[posb];
				}
			}
		}
	}
	// compute the expected values
	for(int t=0; t<length_seq; t++) {
		Score* features=getFeatures(t);
		memset(prob_weight_sum,0, num_gates*sizeof(double));
		if(t==0) {
			int leftState = DUMMY;
			for(int currState=0; currState<num_label; currState++) {
				int currLabel=currState;
				Score prob = (*backward)(currState,t) + ComputeScore(leftState,currState,t) - Partition;
				prob = exp(prob);//t0 trans
				// Trans feauture;
				//        cerr<<(1 + leftState)*num_label + currLabel<<" ";
				m_pModel->grad[(1 + leftState)*num_label + currLabel] -= prob;
				for(int i=0; i<num_gates; i++) {
					// label-based gates weights
					Score out=gates(i,t);
					double prob_out = prob*out;
					//[TP]
					int weightIdx = m_pModel->LocalWeights0 + currState*(num_gates) +i;
					m_pModel->grad[weightIdx] -= prob_out;
					double prob_weight = prob_out*m_pModel->weights[weightIdx]*(1.-out);
					prob_weight_sum[i] +=prob_weight;
				}
			}
		} else {
			// Trans Weights
			for(int leftState=0; leftState<num_states; leftState++)
				for(int currState=num_label; currState<num_states; currState++) {
					int currLabel=currState % num_label;
					if( (leftState % num_label) != (currState / num_label) -1 ) {
						continue;
					}
					if( (t==1) && (leftState >= num_label) ||
					    (t > 1) && (leftState < num_label)) {
						continue;
					}
					Score* features=getFeatures(t);
					Score prob = (*forward)(leftState,t-1) + (*backward)(currState,t) + ComputeScore(leftState,currState,t) - Partition;//@@@
					prob = exp(prob);
					// Trans feauture;
					//          cerr<<(1 + leftState)*num_label + currLabel<<" ";
					m_pModel->grad[(1 + leftState)*num_label + currLabel] -= prob;
				}
			//Local weights
			for(int currState=0; currState<num_states; currState++) {
				Score prob = (*backward)(currState,t) + (*forward)(currState,t) - Partition;//@@@
				prob = exp(prob);
				for(int i=0; i<num_gates; i++) {
					Score out=gates(i,t); // label-based gates weights
					int weightIdx = LocalWeights0 + currState*num_gates +i;
					double prob_out = prob*out;
					m_pModel->grad[weightIdx] -= prob_out;
					//inner parameters
					double prob_weight = prob_out*m_pModel->weights[weightIdx]*(1.-out);
					prob_weight_sum[i] +=prob_weight;
				}
			}
			for(int i=0; i<num_gates; i++) {
				int gradStart = fstart + i*m_pModel->dim_features;
				if(i < num_ps_gates) {
					int gradStart = fstart + i*((dim_ps+1)*window_size+1);
					int pos;
					for(int w=0; w<window_size; w++) {
						for(int j=0; j<dim_ps; j++) {
							pos = w * (dim_dense+1) + j;
							m_pModel->grad[ gradStart ++] -= prob_weight_sum[i]*features[pos];
						}
						m_pModel->grad[ gradStart ++] -= prob_weight_sum[i]*features[ w * (dim_dense+1) + dim_dense];//bias
					}
					m_pModel->grad[ gradStart ++] -= prob_weight_sum[i]*1;//bias
				}
				if(i >= num_ps_gates) {
					int gradGStart = PiInterWeights0 + (i - num_ps_gates) * window_size *2;
					for(int w=0; w<window_size; w++) {
						int pos = w * (dim_dense+1) + i - num_ps_gates + dim_ps;// gate - num_gates < dim_pi;
						m_pModel->grad[ gradGStart ++] -= prob_weight_sum[i]*features[pos];
						int posb = w * (dim_dense + 1) + dim_dense;
						m_pModel->grad[ gradGStart ++] -= prob_weight_sum[i]*features[posb];
					}
				}
			}
		}
	}
	
	delete[] prob_weight_sum;
}
void SEQUENCE::ComputeTestAccuracy()
{
	m_pModel->totalPos += length_seq;
	// comparison
	q3=0;
	for(int t=0; t<length_seq; t++) {
		if(obs_label[t]/2==predicted_label[t]) { //predicted label has been has halfed
			q3++;
			m_pModel->totalCorrect++, m_pModel->apw+=1.0/length_seq;
		}
	}
	//	for(int t=0; t<length_seq;t++)
	//		cout << predicted_label[t] << ":" << obs_label[t] << " ";
	//	cout << endl;
}
void SEQUENCE::ComputeTestAccuracyL8()
{
	m_pModel->totalPos += length_seq;
	// comparison
	q3=0;
	for(int t=0; t<length_seq; t++)
		if(m_pModel->l83[obs_label[t]/2]==m_pModel->l83[predicted_label[t]]) {
			q3++;
			m_pModel->totalCorrect++, m_pModel->apw+=1.0/length_seq;
		}
	//	for(int t=0; t<length_seq;t++)
	//		cout << predicted_label[t] << ":" << obs_label[t] << " ";
	//	cout << endl;
}
void _LBFGS::Report(const vector<double> &theta, int iteration, double objective, double step_length)
{
	if(iteration != 1 && iteration% 100) return;
	for(int i=0; i<m_pModel->num_params; i++)
		m_pModel->weights[i] = theta[i];
	int tc_sum = 0, tp_sum = 0;
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(m_pModel->weights, m_pModel->num_params, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
	m_pModel->totalPos = m_pModel->totalCorrect = 0;
	m_pModel->totalviterbiprob=LogScore_ZERO;
	for(int i=0; i<m_pModel->num_tst; i++) {
		m_pModel->testData[i]->ComputeGates();
		m_pModel->testData[i]->ComputeVi();
		m_pModel->testData[i]->ComputeViterbi();
		m_pModel->testData[i]->ComputeTestAccuracy();
		m_pModel->totalviterbiprob += m_pModel->testData[i]->viterbiprob;
	}
	double norm_w = 0;
	for(int i=0; i<theta.size(); i++)
		norm_w +=theta[i]*theta[i];
	tp_sum = m_pModel->totalPos;
	tc_sum = m_pModel->totalCorrect;
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&m_pModel->totalPos, &tp_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&m_pModel->totalCorrect, &tc_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
#else
	tp_sum=m_pModel->totalPos;
	tc_sum=m_pModel->totalCorrect;
#endif
	if(proc_id==0) {
		cout << endl << "Iteration:  " << iteration << endl;
		cout << " Weight Norm: " << sqrt(norm_w) << endl;
		cout << " Objective: " << objective << endl;
		cout << " test ACC(Viterbi): " << (double) tc_sum/tp_sum <<"   " << tc_sum << "/" << tp_sum << endl;
		string m_file = m_pModel->model_file; // model file path
		m_file+="/model.";
		char buf[100];
		//    jobid = getenv ("JOB_ID");
		sprintf(buf,"p%d-%d.%s",m_pModel->phase,iteration,m_pModel->jobid);
		m_file+=buf;
		cout << m_file << endl;
		ofstream fout(m_file.c_str());
		fout << "num_params: " << m_pModel->num_params << endl;
		fout << "num_gates: " << m_pModel->num_gates << endl;
		fout << "dim_pi: " << m_pModel->dim_pi << endl;
		fout << "dim_ps: " << m_pModel->dim_ps << endl;
		fout << "window_size: " << m_pModel->window_size << endl;
		fout << "dim_features: " << m_pModel->dim_features << endl;
		for(map<string,string>::iterator it=m_pModel->params.begin(); it!=m_pModel->params.end(); it++)
			fout<<it->first<<","<<it->second<<endl;
		fout << "weights: " << endl;
		for(int i=0; i<m_pModel->num_params; i++)
			fout << m_pModel->weights[i] << " ";
		fout << endl;
		fout.close();
	}
	/*
	  m_pModel->totalPos = m_pModel->totalCorrect = 0;
	  for(int i=0;i<m_pModel->num_tst;i++){
	  m_pModel->testData[i]->MAP2();
	  m_pModel->testData[i]->ComputeTestAccuracy();
	  }
	  MPI_Barrier(MPI_COMM_WORLD);
	  MPI_Reduce(&m_pModel->totalPos,&tp_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	  MPI_Reduce(&m_pModel->totalCorrect,&tc_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	  if(proc_id==0){
	cout << " test ACC(MAP2): " << (double) tc_sum/tp_sum <<"   " << tc_sum << "/" << tp_sum << endl;
	  }
	//*/
	/*
	  m_pModel->totalPos = m_pModel->totalCorrect = 0;
	  for(int i=0;i<m_pModel->num_tst;i++){
	  m_pModel->testData[i]->onebest();
	  m_pModel->testData[i]->ComputeTestAccuracy();
	  }
	  MPI_Barrier(MPI_COMM_WORLD);
	  MPI_Reduce(&m_pModel->totalPos,&tp_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	  MPI_Reduce(&m_pModel->totalCorrect,&tc_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	  if(proc_id==0){
	cout << " test ACC(onebest): " << (double) tc_sum/tp_sum <<"   " << tc_sum << "/" << tp_sum << endl;
	  }
	//*/
	m_pModel->totalPos = m_pModel->totalCorrect = 0;
	m_pModel->totalmapprob=LogScore_ZERO;
	for(int i=0; i<m_pModel->num_tst; i++) {
		m_pModel->testData[i]->MAP1();
		m_pModel->testData[i]->ComputeTestAccuracy();
		m_pModel->totalmapprob += m_pModel->testData[i]->mapprob;
	}
	tp_sum = m_pModel->totalPos;
	tc_sum = m_pModel->totalCorrect;
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&m_pModel->totalPos,&tp_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&m_pModel->totalCorrect,&tc_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#else
	tp_sum=m_pModel->totalPos;
	tc_sum=m_pModel->totalCorrect;
#endif
	if(proc_id==0) {
		cout << " test ACC(MAP1): " << (double) tc_sum/tp_sum <<"   " << tc_sum << "/" << tp_sum << endl;
	}
	//output the detail of classification result.
	if(!(iteration  % 50) && 0) {
		char seq_file[500];
		char buf[500];
		sprintf(seq_file,"%s/ens-%s-%s-proc-%d",m_pModel->workdir.c_str(),m_pModel->params["ACT"].c_str(),m_pModel->jobid,proc_id);
		ofstream seqsout(seq_file);
		for(int si=0; si<m_pModel->testData.size(); si++) {
			string lineout;
			for(int i=0; i<m_pModel->testData[si]->length_seq; i++) {
				sprintf(buf,"%d",m_pModel->testData[si]->obs_label[i]/2);
				lineout+=buf;
			}
			lineout+=" ";
			for(int i=0; i<m_pModel->testData[si]->length_seq; i++) {
				sprintf(buf,"%d",m_pModel->testData[si]->predicted_label[i]/2);
				lineout+=buf;
			}
			sprintf(buf,"%f ",((float)m_pModel->testData[si]->q3)/m_pModel->testData[si]->length_seq);
			lineout=buf+lineout;
			sprintf(buf,"%5d ",m_pModel->testData[si]->length_seq);
			lineout=buf+lineout;
			lineout+=" ";
			for(int i=0; i<m_pModel->testData[si]->length_seq; i++) {
				for(int l=0; l<m_pModel->num_label; l++) {
					sprintf(buf,"%.4f ",exp(m_pModel->testData[si]->predicted_allstates[i*m_pModel->num_label+l]));
					lineout+=buf;
				}
			}
			seqsout << lineout<<endl;
			//    sprintf(buf,"",
		}
	}
	if(!(iteration  % 500) || iteration==1 ) {
		char seq_file[500];
		char buf[500];
		char resultname[500];
		//for some bug in MAP1, we should recalculate predicted_labels
		// sprintf(seq_file,"result.l8",m_pModel->workdir.c_str(),m_pModel->params["ACT"].c_str(),m_pModel->jobid,proc_id);
		if(m_pModel->params["RESULT"]=="")
			sprintf(seq_file,"result.eie.%s",m_pModel->jobid);
		else
			sprintf(seq_file,"%s",m_pModel->params["RESULT"].c_str());
		ofstream seqsout(seq_file);
		for(int si=0; si<m_pModel->testData.size(); si++) {
			string lineout;
			for(int i=0; i<m_pModel->testData[si]->length_seq; i++) {
				sprintf(buf,"%d",m_pModel->testData[si]->obs_label[i]/2);
				lineout+=buf;
			}
			lineout+=" ";
			for(int i=0; i<m_pModel->testData[si]->length_seq; i++) {
				sprintf(buf,"%d",m_pModel->testData[si]->predicted_label[i]);
				lineout+=buf;
			}
			sprintf(buf,"%f %.4f %.4f %.4f ",((float)m_pModel->testData[si]->q3)/m_pModel->testData[si]->length_seq, -1, -1, -1);
			lineout=buf+lineout;
			sprintf(buf,"%5d ",m_pModel->testData[si]->length_seq);
			lineout=buf+lineout;
			lineout+=" ";
			for(int i=0; i<m_pModel->testData[si]->length_seq; i++) {
				for(int l=0; l<m_pModel->num_label/2; l++) { // num_label has been doubled at init.
					sprintf(buf,"%.4f ",exp(m_pModel->testData[si]->predicted_allstates[i*m_pModel->num_label+l*2]));
					lineout+=buf;
				}
			}
			seqsout << lineout<<endl;
		}
		seqsout.close();
	}
	m_pModel->totalPos = m_pModel->totalCorrect = 0;
	for(int i=0; i<m_pModel->num_tst; i++) {
		//    m_pModel->testData[i]->MAP1();
		m_pModel->testData[i]->ComputeTestAccuracyL8();
	}
	tp_sum = m_pModel->totalPos;
	tc_sum = m_pModel->totalCorrect;
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&m_pModel->totalPos,&tp_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&m_pModel->totalCorrect,&tc_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#else
	tp_sum=m_pModel->totalPos;
	tc_sum=m_pModel->totalCorrect;
#endif
	if(proc_id==0) {
		cout << " test ACC(MAP1-L8to3): " << (double) tc_sum/tp_sum <<"   " << tc_sum << "/" << tp_sum << endl;
	}
	/*
	  m_pModel->totalPos = m_pModel->totalCorrect = 0;
	  for(int i=0;i<m_pModel->num_tst;i++){
	  m_pModel->testData[i]->onebest();
	  m_pModel->testData[i]->ComputeTestAccuracy();
	  }
	  MPI_Barrier(MPI_COMM_WORLD);
	  MPI_Reduce(&m_pModel->totalPos,&tp_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	  MPI_Reduce(&m_pModel->totalCorrect,&tc_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	  if(proc_id==0){
	cout << " test ACC(onebest): " << (double) tc_sum/tp_sum <<"   " << tc_sum << "/" << tp_sum << endl;
	  }
	*/
	m_pModel->totalviterbiprob=LogScore_ZERO;
	m_pModel->totalPos = m_pModel->totalCorrect = 0;
	for(int i=0; i<m_pModel->num_data; i++) {
		m_pModel->trainData[i]->ComputeGates();
		m_pModel->trainData[i]->ComputeVi();
		m_pModel->trainData[i]->ComputeViterbi();
		m_pModel->trainData[i]->ComputeTestAccuracy();
		m_pModel->totalviterbiprob += m_pModel->trainData[i]->viterbiprob;
	}
	tp_sum = m_pModel->totalPos;
	tc_sum = m_pModel->totalCorrect;
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&m_pModel->totalPos,&tp_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&m_pModel->totalCorrect,&tc_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#else
		tp_sum=m_pModel->totalPos;
		tc_sum=m_pModel->totalCorrect;
		
#endif
	if(proc_id==0)
		cout <<" train ACC: " << (double) tc_sum/tp_sum << "   " << tc_sum << "/" << tp_sum << endl;
}
void _LBFGS::ComputeGradient(vector<double>&g, const vector<double> &x, bool bCalculateGate)
{
	for(int i=0; i<m_pModel->num_params; i++)
		m_pModel->weights[i] = x[i];
	//weights[num_params-1] = 1;
	double _norm = 0;
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(m_pModel->weights, m_pModel->num_params, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
	for(int i=0; i<m_pModel->num_params; i++)
		m_pModel->grad[i]=0;
	for(int i=0; i<m_pModel->num_data; i++)
		m_pModel->trainData[i]->ComputeGradient(bCalculateGate);
	//for(int i=0;i<num_params;i++) g[i] = grad[i];
	for(int i=0; i<m_pModel->num_params; i++) {
		if(i<m_pModel->PsInterWeights0)//num_states*(m_pModel->num_states+m_pModel->num_gates + 1))
			m_pModel->grad[i] = -m_pModel->grad[i]+m_pModel->weights[i]*m_pModel->reg[i]*2; //best:0.04,regularizer ,zy said
		else
			m_pModel->grad[i] = -m_pModel->grad[i]+m_pModel->weights[i]*m_pModel->reg[i]*2; //best:0.02
	}
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(m_pModel->grad, m_pModel->grad_sum, m_pModel->num_params, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(m_pModel->grad_sum, m_pModel->num_params, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
	for(int i=0; i<m_pModel->num_params; i++)
		m_pModel->grad_sum[i]=m_pModel->grad[i] ;
#endif
	for(int i=0; i<m_pModel->num_params; i++) {
		g[i] = m_pModel->grad_sum[i];
		_norm +=g[i]*g[i];
	}
	if(!proc_id)
		cerr << "Norm of Gradient: " << sqrt(_norm) << endl;
}
double _LBFGS::ComputeFunction(const vector<double> &x)
{
	//    cerr<<"x size="<<x.size()<<"\n";
	// cerr<<"num params="<<m_pModel->num_params<<"\n";
	//  if(!proc_id)
	for(int i=0; i<m_pModel->num_params; i++) {
		m_pModel->weights[i] = x[i];
		//      cerr<<x[i]<<" ";
	}
	//  cerr<<endl;
	//weights[num_params-1] = 1;
	//  cerr<<"step0cf"<<proc_id<<endl;
	//cerr<<"num_params"<<
	//    m_pModel->num_params<<endl;
	//  cerr<<"mpidouble "<<sizeof(MPI_DOUBLE)<<endl;
	//  cerr<<"sizeofweights"<<m_pModel->weights[m_pModel->num_params-1]<<endl;
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	//cerr<<"step01cf"<<proc_id<<endl;
	MPI_Bcast(m_pModel->weights,m_pModel->num_params , MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
	//  cerr<<"step02cf"<<proc_id<<endl;
	// MPI_Barrier(MPI_COMM_WORLD);
	//  cerr<<"cf"<<proc_id<<endl;
	double obj = 0, obj_sum = 0;
	for(int i=0; i<m_pModel->num_data; i++)
		obj += m_pModel->trainData[i]->Obj();
	obj = -obj;
	for(int i=0; i<x.size(); i++) {
		if(i<m_pModel->PsInterWeights0)//num_states*(m_pModel->num_states+m_pModel->num_gates+1))
			obj += m_pModel->weights[i]*m_pModel->weights[i]*m_pModel->reg[i];
		else
			obj += m_pModel->weights[i]*m_pModel->weights[i]*m_pModel->reg[i];
	}
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&obj, &obj_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Bcast(&obj_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else 
	obj_sum=obj;
#endif

	//  cerr<<"step2cf"<<proc_id<<endl;
	return obj_sum;
}
void _LBFGS::Report(const string &s)
{
	if(!proc_id) cerr << s << endl;
}
void bCNF_Model::SetSeed()
{
	unsigned int randomSeed=0;
	ifstream in("/dev/urandom",ios::in);
	in.read((char*)&randomSeed, sizeof(unsigned)/sizeof(char));
	in.close();
	//unsigned id=getpid();
	randomSeed=randomSeed*randomSeed;//+id*id;
	//we can set the random seed at only the main function
	srand48(randomSeed);
	srand(randomSeed);
}
void bCNF_Model::SetParameters(int argc,char **argv)
{
	phase=0;
	jobid=new char[1000];
	//jobid = getenv ("JOB_ID");
	if(proc_id==0) {
		char* jobid_env = getenv ("JOB_ID");
		if(jobid_env==NULL) {
			sprintf(jobid,"%d",rand() % 10000);
		}
		else{
			sprintf(jobid,"%s",jobid_env);
		}			
	}
	
	if(proc_id==0)cerr<<"$JOB_ID "<<jobid<<" "<<strlen(jobid)<<endl;
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(jobid,1000, MPI_CHAR, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	if(proc_id==0)cerr<<"$JOB_ID bcast "<<jobid<<" "<<strlen(jobid)<<endl;
	params["CONF"]="CNF.conf";
	string parakey,paraval;
	for(int i=0; i<floor((double)(argc-1)/2); i++) {
		parakey.assign(argv[i*2+1]);
		paraval.assign(argv[i*2+2]);
		params[parakey]=paraval;
	}
	ifstream param_in(params["CONF"].c_str());
	if(param_in.good()){
	while(!param_in.eof()) {
		param_in>>parakey;
		param_in>>paraval;
		if(params[parakey].length()==0)	params[parakey]=paraval;
	}
	
	}
	//set learning rate, regularizaton lamda_v and lamda_w
	SetSeed();
	//learning_rate = 1;
	string modellist;
	modellist=params["MODELLIST"];
	nModel=second_order;
	if(modellist!="") {
		ifstream ifmodellist(modellist.c_str());
		while(!ifmodellist.eof()) {
			string modelfile;
			string testfile;
			string trainfile;
			ifmodellist>>modelfile;
			if(!ifmodellist.eof()) ifmodellist>>trainfile;
			if(!ifmodellist.eof()) ifmodellist>>testfile;
			models.push_back(modelfile);
			testfiles.push_back(testfile);
			trainfiles.push_back(trainfile);
		}
		//    models.pop_back();
		if(proc_id==0)cerr<<models.size()<<" models in total.\n";
	}
	l83[0]= 0;
	l83[1]= 0;
	l83[4]= 1;
	l83[3]= 1;
	l83[7]= 2;
	l83[6]= 2;
	l83[5]= 2;
	l83[2]= 2;
	double lamda_v = atof(params["LAMDA_V"].c_str())/num_procs;
	double lamda_w = atof(params["LAMDA_W"].c_str())/num_procs;
	bien=atoi(params["BIEN"].c_str());//how many aacid far from the trans point are considered as begin or end.
	featmask=atoi(params["FEATMASK"].c_str());
	window_size = atoi(params["WINDOW_LEN"].c_str()); //best 13
	trndata=params["TRAIN"];
	tstdata=params["TEST"];//.assign(argv[2]);
	workdir=params["OUTDIR"];//.assign(argv[3]);
	num_label=atoi(params["NUM_LABEL"].c_str());
	num_label=num_label*2;
	//workdir=workdir+jobid;
	model_file=workdir;
	mkdir(workdir.c_str(), S_IRWXU | S_IRWXG);
	if(proc_id==0)cerr<<"param: "<<trndata<<" "<<tstdata<<" "<<workdir<<"\n";
	//  epoch = 5;
	num_gates = atoi(params["NUM_GATES"].c_str()); //best 20
	//  dim_one_pos = atoi(params["FEAT_DIM"].c_str());//dimension of feature vectors+1
	//[TP]
	dim_ps = atoi(params["DIM_PS"].c_str());//dimension of position specific feature
	dim_pi = atoi(params["DIM_PI"].c_str());//dimension of position independent feature
	dim_dense=dim_ps+dim_pi;
	dim_one_pos_sparse = atoi(params["FEAT_DIM_SPARSE"].c_str());//dimension of the sparse feature
	dim_one_pos = dim_dense;
	//  dim_dense=dim_ps+dim_;
	dim_features = window_size*(dim_dense+1)+dim_one_pos_sparse+1;//To avoid increase the dimension much, we record the sparse part of the features.
	//[TP]
	num_ps_gates=num_gates;
	num_pi_gates=dim_pi;
	//num_pi_gates=10;// for debug.
	num_gates=num_ps_gates+num_pi_gates;
	//num_label=3; //8-class prediction;
	num_states=num_label*(num_label+1); //number of  DUMMY.label plus number of label.label
	LocalWeights0 = (1 + num_states) * num_label;
	PsInterWeights0 = LocalWeights0 + num_states*num_gates;
	PiInterWeights0 = PsInterWeights0 + (window_size*(dim_ps+1)+1)*num_ps_gates;
	num_params = PiInterWeights0 + window_size*num_pi_gates*2 ;
	//  num_params = num_label*(1 + num_states); // Transition weights
	//num_params +=num_states*num_gates + (window_size*(dim_ps+1)+1)*num_ps_gates ;// the gates given in the parameters  are connected to the PSSM features, include local bias and global bias
	//num_params += window_size*num_pi_gates*2;//Every one dimension of extra features in a window is connected to one extra gate.
	num_params += 1;//Add a null gate to present the offset.
	//  cerr<<"num_params:"<<num_params<<endl;
	//cerr<<"num_params:"<<num_params<<endl;
	if(proc_id==0){
	cerr << "num_states: " << num_states << endl;
	cerr << "num_label: " << num_label << endl;
	cerr << "num_params: " << num_params << endl;
	cerr << "num_gates: " << num_gates << endl;
	cerr << "num_ps_gates: " << num_ps_gates << endl;
	cerr << "num_pi_gates: " << num_pi_gates << endl;
	cerr << "window_size: " << window_size << endl;
	cerr << "dim_features: " << dim_features << endl;
	cerr << "dim_one_pos: " << dim_one_pos << endl;
	
	}
	weights = new double[num_params];
	grad = new double[num_params];
	grad_sum = new double[num_params];
	if(!proc_id) {
		cerr <<"num_params = " << num_params << endl;
		for(int i=0; i<num_params; i++)
			weights[i]=(drand48()-drand48())/10;
		int init_from_file = 0;
		if(params["RESUME"]!="") {
			ifstream fin(params["RESUME"].c_str());
			if (fin.fail()) {
				//        cerr << "Error opening resume model\n";
				exit(1);
			} else {
				cerr <<"open model success\n";
			}
			cerr << "Loading weights from " <<params["RESUME"] << endl;
			string s="";
			while(s!="weights:") {
				fin>>s;
			}
			for(int i=0; i<num_params; i++) {
				fin >> weights[i];
			}
			fin.close();
		}
	}
	//???
	reg = new double[num_params];
	for(int i=0; i<num_params; i++)
		reg[i] = lamda_v;
	for(int i=0; i<PsInterWeights0; i++)
		reg[i] = lamda_w;
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(weights, num_params, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(reg, num_params, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	//  if(jobid!=NULL)
	//  MPI_Bcast(jobid,strlen(jobid) , MPI_CHAR, 0, MPI_COMM_WORLD);
	//MPI_Barrier(MPI_COMM_WORLD);
}
void bCNF_Model::Initialize(int argc,char** argv)
{
	//	model_file = model_dir + "Model/model.g";
	SetParameters(argc,argv);
	trainData.clear();
	testData.clear();
	LoadData();
	for(int i=0; i<num_data; i++) {
		//    cerr<<"p"<<proc_id<<"mtr"<<i<<",";
		trainData[i]->makeFeatures();
	}
	for(int i=0; i<num_tst; i++) {
		//cerr<<"p"<<proc_id<<"mts"<<i<<",";
		testData[i]->makeFeatures();
	}
}
//int bypass;
void bCNF_Model::LoadData()
{
	ifstream trn_in(trndata.c_str());//"/home/zywang/00workshop/09datasets/test1/combined.train.input");//create a input stream from file data2 for training
	if(trn_in.good()) {
		trn_in >> num_data;
	} else {
		num_data=0;
	}
	//  num_data=10;
	vector<SEQUENCE*> DATA;
	vector<SEQUENCE*> tstDATA;
	int length_seq;
	double tmp;
	for(int i=0; i<num_data; i++) {
		// construct a new sequence
		trn_in >> length_seq;//Read the length of the sequence.
		//	  cerr<< "the seq length is "<<length_seq;
		SEQUENCE *seq = new SEQUENCE(length_seq, this);
		for(int j=0; j<length_seq; j++) {
			for(int k=0; k<dim_one_pos; k++) {
				trn_in >> seq->obs_feature[j][k];
			}
			seq->obs_feature[j][dim_one_pos] = 1;
		}
		
		for(int j=0; j<length_seq; j++) {
			trn_in >> seq->obs_label[j];
		}
		
		if(i%num_procs != proc_id) {
			delete seq;
			continue;
		}
		seq->modifilabels(bien);
		seq->modififeatures(featmask);
		DATA.push_back(seq);
	}
	trn_in.close();
	num_data = DATA.size();
	if(proc_id==0)cerr<< "Training data"<< num_data <<" loaded"<<endl;
	if(params["ACT"] != "") {
    if(params["TEST"]==""){
    }else{
      ifstream tst_in(tstdata.c_str());
      tst_in >> num_tst;
      //    num_tst=10;
      for(int i=0; i<num_tst; i++) {
        // construct a new sequence
        tst_in >> length_seq;
        SEQUENCE *seq = new SEQUENCE(length_seq, this);
        for(int j=0; j<length_seq; j++) {
          for(int k=0; k<dim_one_pos; k++) {
            tst_in >> seq->obs_feature[j][k];
          }
          seq->obs_feature[j][dim_one_pos] = 1;
        }
			
        for(int j=0; j<length_seq; j++) {
          tst_in >> seq->obs_label[j];
        }
			
        if(i%num_procs != proc_id) {
          delete seq;
          continue;
        }
        seq->modififeatures(featmask);
        seq->modifilabels(bien);
        tstDATA.push_back(seq);
      }
      tst_in.close();
    }
    
		num_tst = tstDATA.size();
		if(proc_id==0)cerr << proc_id << " " << num_tst << endl;
		for(int i=0; i<num_tst; i++) {
			testData.push_back(tstDATA[i]);
		}
		//num_data = num_data - num_tst;
		for(int i=0; i<num_data; i++) {
			trainData.push_back(DATA[i]);
		}
	} else {
		vector<int> shuffle;
		num_data = DATA.size();
		for(int i=0; i<num_data; i++)
			shuffle.push_back(num_data-1-i);
		random_shuffle(shuffle.begin(),shuffle.end());
		num_tst = num_data/5;
		for(int i=0; i<num_tst; i++) {
			testData.push_back(DATA[shuffle[i]]);
		}
		num_data = num_data - num_tst;
		for(int i=0; i<num_data; i++) {
			trainData.push_back(DATA[shuffle[i+num_tst]]);
		}
	}
	//Show some labels and features
	if(proc_id==0){
	cerr<<"some training data\n";
	cerr<<trainData[0]->obs_label[0]<<" "<<trainData[0]->obs_feature[0][0]<<" "<<trainData[0]->obs_feature[0][1]<<endl;
	cerr<<trainData[0]->obs_label[1]<<" "<<trainData[0]->obs_feature[1][0]<<" "<<trainData[0]->obs_feature[1][1]<<endl;
	
	}
}
void bCNF_Model::LoadRetrainData()
{
	ifstream trn_in(retrndata.c_str());//"/home/zywang/00workshop/09datasets/test1/combined.train.input");//create a input stream from file data2 for training
	if(trn_in.good()) {
		trn_in >> num_data;
	} else {
		num_data=0;
	}
	//    num_data=1000;
	vector<SEQUENCE*> DATA;
	vector<SEQUENCE*> tstDATA;
	int length_seq;
	double tmp;
	for(int i=0; i<num_data; i++) {
		// construct a new sequence
		trn_in >> length_seq;//Read the length of the sequence.
		//	  cerr<< "the seq length is "<<length_seq;
		SEQUENCE *seq = new SEQUENCE(length_seq, this);
		for(int j=0; j<length_seq; j++) {
			for(int k=0; k<dim_one_pos; k++) {
				trn_in >> seq->obs_feature[j][k];
			}
			seq->obs_feature[j][dim_one_pos] = 1;
		}
		for(int j=0; j<length_seq; j++) {
			trn_in >> seq->obs_label[j];
		}
		DATA.push_back(seq);
	}
	trn_in.close();
	num_data = DATA.size();
	if(proc_id==0)cerr<< "Training data"<< num_data <<" loaded"<<endl;
	ifstream tst_in(retstdata.c_str());
	tst_in >> num_tst;
	for(int i=0; i<num_tst; i++) {
		// construct a new sequence
		tst_in >> length_seq;
		SEQUENCE *seq = new SEQUENCE(length_seq, this);
		for(int j=0; j<length_seq; j++) {
			for(int k=0; k<dim_one_pos; k++) {
				tst_in >> seq->obs_feature[j][k];
			}
			seq->obs_feature[j][dim_one_pos] = 1;
		}
		for(int j=0; j<length_seq; j++) {
			tst_in >> seq->obs_label[j];
		}
		tstDATA.push_back(seq);
	}
	tst_in.close();
	num_tst = tstDATA.size();
	if(proc_id==0)cerr << proc_id << " " << num_tst << endl;
	for(int i=0; i<num_tst; i++) {
		testData.push_back(tstDATA[i]);
	}
	//num_data = num_data - num_tst;
	for(int i=0; i<num_data; i++) {
		trainData.push_back(DATA[i]);
	}
}
void bCNF_Model::ReloadData(string testfile,string trainfile)
{
	ifstream trn_in(trainfile.c_str());//"/home/zywang/00workshop/09datasets/test1/combined.train.input");//create a input stream from file data2 for training
	if(trn_in.good()) {
		trn_in >> num_data;
	} else {
		num_data=0;
	}
	//only for test
	num_data=10;
	vector<SEQUENCE*> DATA;
	vector<SEQUENCE*> tstDATA;
	double tmp;
	for(int i=0; i<num_data; i++) {
		if(params["ACT"] != "RETRAIN") break;
		int length_seq;
		// construct a new sequence
		trn_in >> length_seq;//Read the length of the sequence.
		//	  cerr<< "the seq length is "<<length_seq;
		SEQUENCE *seq = new SEQUENCE(length_seq, this);
		for(int j=0; j<length_seq; j++) {
			for(int k=0; k<dim_one_pos; k++) {
				trn_in >> seq->obs_feature[j][k];
			}
			seq->obs_feature[j][dim_one_pos] = 1;
		}
		for(int j=0; j<length_seq; j++) {
			trn_in >> seq->obs_label[j];
		}
		if(i%num_procs != proc_id) {
			delete seq;
			continue;
		}
		DATA.push_back(seq);
	}
	trn_in.close();
	num_data = DATA.size();
	if(proc_id==0)cerr<< "Training data"<< num_data <<" loaded"<<endl;
	ifstream tst_in(testfile.c_str());//"/home/zywang/00workshop/09datasets/test1/combined.test.input");//test
	tst_in >> num_tst;
	//only for test
	num_tst=10;
	for(int i=0; i<num_tst; i++) {
		int length_seq;
		// construct a new sequence
		tst_in >> length_seq;
		SEQUENCE *seq = new SEQUENCE(length_seq, this);
		for(int j=0; j<length_seq; j++) {
			for(int k=0; k<dim_one_pos; k++) {
				tst_in >> seq->obs_feature[j][k];
			}
			seq->obs_feature[j][dim_one_pos] = 1;
		}
		for(int j=0; j<length_seq; j++) {
			tst_in >> seq->obs_label[j];
		}
		vector<int> bielabel(length_seq);
		for(int j=0; j<length_seq; j++) {
			if(j+1<length_seq && seq->obs_label[j]!=seq->obs_label[j+1]) {
				bielabel[j] = seq->obs_label[j]*3 + 2;
			}
			if(j-1>=0 && seq->obs_label[j]!=seq->obs_label[j-1]) {
				bielabel[j] = seq->obs_label[j]*3 + 0;
			}
			bielabel[j] = seq->obs_label[j]*3 + 1;
		}
		for(int j=0; j<length_seq; j++) {
			seq->obs_label[j]=bielabel[j];
		}
		if(i%num_procs != proc_id) {
			delete seq;
			continue;
		}
		tstDATA.push_back(seq);
	}
	tst_in.close();
	num_tst = tstDATA.size();
	if(proc_id==0)cerr << proc_id << " " << num_tst << endl;
	//Other than push_back in loaddata, replace them.
	if(num_tst!=testData.size()) {
		if(proc_id==0)cerr<<"Test sets sizes are different\n";
	}
	for(int i=0; i<num_tst; i++) {
		//    testData.push_back(tstDATA[i]);
		if(testData[i]->length_seq != tstDATA[i]->length_seq) {
			if(proc_id==0)cerr<<"Test seq "<<i<<","<<testData[i]->length_seq<<","<<tstDATA[i]->length_seq<<" lengths are different.\n";
		}
		int df = dim_features*testData[i]->length_seq;
		delete  testData[i]->_features;
		testData[i]->_features = new Score[df];
		for(int ia=0; ia<testData[i]->length_seq; ia++) {
			delete testData[i]->obs_feature[ia];
			testData[i]->obs_feature[ia] = new Score[dim_one_pos+1];
			memcpy(testData[i]->obs_feature[ia], tstDATA[i]->obs_feature[ia], sizeof(Score)*(dim_one_pos+1));
		}
	}
	if(num_data!=trainData.size()) {
		if(proc_id==0)cerr<<"Train sets sizes are different\n";
	}
	for(int i=0; i<num_data; i++) {
		//    trainData.push_back(DATA[i]);
		if(params["ACT"] != "RETRAIN") break;
		if(trainData[i]->length_seq != DATA[i]->length_seq) {
			if(proc_id==0)cerr<<"Test seq "<<i<<","<<trainData[i]->length_seq<<","<<DATA[i]->length_seq<<" lengths are different.\n";
		}
		int df = dim_features*trainData[i]->length_seq;
		delete  trainData[i]->_features;
		trainData[i]->_features = new Score[df];
		for(int ia=0; ia<trainData[i]->length_seq; ia++) {
			delete trainData[i]->obs_feature[ia];
			trainData[i]->obs_feature[ia] = new Score[dim_one_pos+1];
			memcpy(trainData[i]->obs_feature[ia], DATA[i]->obs_feature[ia], sizeof(Score)*(dim_one_pos+1));
		}
	}
}
void bCNF_Model::Ensemble_start()
{
	if(proc_id==0)cerr<<"Begin ensemble prediction.\n";
	for(int kk=0; kk<models.size(); kk++) {
#ifdef _MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		if(!proc_id) {
			if(proc_id==0)cerr<<"The "<<kk<<" model\n";
		}
		string s;
		vector<string> ps;
		ifstream ifmodel(models[kk].c_str());
		while(s!="weights:") {
			ifmodel>>s;
			ps.push_back(s);
		}
		if(ps.size() %2) { // Complete the params mapping
			ps.push_back("");
		}
		for(int i=0; i<ps.size(); i=i+2) {
			params[ps[i]]=ps[i+1];
		}
		int old_num_params=num_params;
		num_params=atoi(params["num_params:"].c_str());
		dim_ps=atoi(params["dim_ps:"].c_str());
		dim_pi=atoi(params["dim_pi:"].c_str());
		num_gates=atoi(params["num_gates:"].c_str());
		dim_dense=dim_ps+dim_pi;
		dim_one_pos = dim_dense;
		num_ps_gates=num_gates;
		num_pi_gates=dim_pi;
		//num_pi_gates=10;// for debug.
		num_gates=num_ps_gates+num_pi_gates;
		num_states=num_label*(num_label+1); //number of  DUMMY.label plus number of label.label
		LocalWeights0 = (1 + num_states) * num_label;
		PsInterWeights0 = LocalWeights0 + num_states*num_gates;
		PiInterWeights0 = PsInterWeights0 + (window_size*(dim_ps+1)+1)*num_ps_gates;
		num_params = PiInterWeights0 + window_size*num_pi_gates*2 ;
		num_params += 1;//Every one dimension of extra features in a window is connected to one extra gate.
		//    num_params += 1;//Add a null gate to present the offset.
		dim_features = window_size*(dim_dense+1)+1;//To avoid incre
		cerr <<num_params<<","<<dim_ps<<","<<dim_pi<<","<<num_gates<<","<<dim_features <<"," <<dim_one_pos<< endl;
		if(weights)
			delete weights;
		weights=new double[num_params];
		window_size=atoi(params["window_size:"].c_str());
		int pread=0;
		while(ifmodel.good()) {
			ifmodel >> weights[pread];
			pread++;
		}
		if(pread-1!=num_params) {
			cerr<<"read weights error!"<<pread-1<<" "<<num_params<<"\n";
		}
		ifmodel.close();
		cerr <<"Finished reading model "<< models[kk] <<".\n";
		cerr <<"Reload train &test sets for this model."<<testfiles[kk]<<" "<<trainfiles[kk]<<"\n";
		if(params["ACT"]!= "RETRAIN" && kk>=1 && testfiles[kk]!=testfiles[kk-1]) ReloadData(testfiles[kk],trainfiles[kk]);
		if(params["ACT"] == "RETRAIN")
			for(int i=0; i<num_data; i++) {
				trainData[i]->makeFeatures();
				trainData[i]->ComputeGates();
				trainData[i]->ComputeVi();
				trainData[i]->MAP1();
				trainData[i]->allresults.push_back(trainData[i]->predicted_allstates);
				trainData[i]->predicted_allstates=new Score[trainData[i]->length_seq*num_states];
			}
		cerr <<"Begin test data prediction "<<proc_id<<" :"<< models[kk] <<".\n";
		cerr.flush();
		cerr<<"Reload finished.\n";
		for(int i=0; i<num_tst; i++) {
			// cerr<<i<<".";
			testData[i]->makeFeatures();
			testData[i]->ComputeGates();
			testData[i]->ComputeVi();
			testData[i]->MAP1();
			testData[i]->allresults.push_back(testData[i]->predicted_allstates);
			testData[i]->predicted_allstates=new Score[testData[i]->length_seq*num_states];
		}
	}
	cerr<<"end of predicting by every models=================\n";
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	//End of all the models and voting.
	//================================================
	totalCorrect=0;
	totalPos=0;
	if(params["ACT"] == "RETRAIN")
		for(int i=0; i<num_data; i++) {
			trainData[i]->Voting();
			trainData[i]->ComputeTestAccuracy();
			//cout<<"Proc "<<proc_id<<" train_seq "<<i<<" q3 "<< trainData[i]->q3/trainData[i]->length_seq <<endl;
			char filename[1000];
			sprintf(filename,"/train,%s,%d,proc,%d.seq",jobid,i,proc_id);
			ofstream ofseq(filename);
			ofseq<<(trainData[i]->length_seq)<<endl;
			for(int j=0; j<trainData[i]->length_seq; j++) {
				for(int k=0; k<models.size(); k++) {
					for(int ks=0; ks<num_states; ks++) {
						ofseq<<trainData[i]->allresults[k][j*num_states+ks]<<" ";
					}
					ofseq<<endl;
				}
			}
			for(int j=0; j<trainData[i]->length_seq; j++) {
				ofseq<<trainData[i]->obs_label[j]<<endl;
			}
			ofseq.close();
		}
	totalCorrect=0;
	totalPos=0;
	char seq_file[500];
	char buf[500];
	sprintf(seq_file,"%s/ens-proc-%d",workdir.c_str(),proc_id);
	ofstream seqsout(seq_file);
	cerr<<"Begin test data, totally "<<num_tst<<endl;
	for(int si=0; si<num_tst; si++) {
		testData[si]->Voting();
		testData[si]->ComputeTestAccuracy();
		cerr<<"Proc "<<proc_id<<" test_seq "<<si<<" q3 "<<((float)testData[si]->q3)/testData[si]->length_seq<<endl;
		char filename[1000];
		sprintf(filename,"%s/test,%d,proc,%d.seq",workdir.c_str(),si,proc_id);
		ofstream ofseq(filename);
		ofseq<<(testData[si]->length_seq)<<endl;
		for(int j=0; j<testData[si]->length_seq; j++) {
			for(int km=0; km<models.size(); km++) {
				for(int k=0; k<num_states; k++) {
					Score *p=testData[si]->allresults[km];
					ofseq<<p[j*num_states+k]<<" ";
				}
			}
			ofseq<<endl;
		}
		for(int j=0; j<testData[si]->length_seq; j++) {
			ofseq<<testData[si]->obs_label[j]/2<<endl;
		}
		ofseq.close();
		string lineout;
		for(int i=0; i<testData[si]->length_seq; i++) {
			sprintf(buf,"%d",testData[si]->obs_label[i]/2);
			lineout+=buf;
		}
		lineout+=" ";
		for(int i=0; i<testData[si]->length_seq; i++) {
			sprintf(buf,"%d",testData[si]->predicted_label[i]/2);
			lineout+=buf;
		}
		sprintf(buf,"%f ",((float)testData[si]->q3)/testData[si]->length_seq);
		lineout=buf+lineout;
		sprintf(buf,"%5d ",testData[si]->length_seq);
		lineout=buf+lineout;
		seqsout << lineout<<endl;
	}
	seqsout.close();
}
void bCNF_Model::Double_train() //==================================
{
	int tc_sum = 0, tp_sum = 0;
	//Compute MAP1
	phase=1;
	for(int i=0; i<num_tst; i++) {
		testData[i]->MAP1();
		testData[i]->allresults.push_back(testData[i]->predicted_allstates);
		testData[i]->ComputeTestAccuracy();
	}
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&totalPos,&tp_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&totalCorrect,&tc_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#else
	tp_sum=totalPos;
	tc_sum=totalCorrect;
#endif
	if(proc_id==0) {
		cout << " test ACC(MAP1): " << (double) tc_sum/tp_sum <<"   " << tc_sum << "/" << tp_sum << endl;
	}
	//Compute MAP1 on train data sets
	totalPos = totalCorrect = 0;
	for(int i=0; i<num_data; i++) {
		trainData[i]->MAP1();
		trainData[i]->ComputeTestAccuracy();
		trainData[i]->allresults.push_back(trainData[i]->predicted_allstates);
		//    trainData[i]->allresults.push_back(trainData[i]->predicted_allstates);
	}
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Reduce(&totalPos,&tp_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&totalCorrect,&tc_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
#else
	tp_sum=totalPos;
	tc_sum=totalCorrect;
#endif
	if(proc_id==0) {
		cout << " train ACC(MAP1): " << (double) tc_sum/tp_sum <<"   " << tc_sum << "/" << tp_sum << endl;
	}
	//Re-build the features of every train data;
	//int old_dim_one_pos=dim_one_pos;
	int old_dim_one_pos=0;
	//  old_dim_features=dim_dense;
	int resultsize= trainData[0]->allresults.size();
	int extrafeat = resultsize * num_label;
	if(0) {
		dim_pi=dim_pi + extrafeat;
		//dim_ps remain;
		dim_dense=dim_ps+dim_pi;
		old_dim_one_pos=dim_one_pos;
		//num_ps_gates=0; remains;
		num_pi_gates=dim_pi;
		num_gates=num_ps_gates+num_pi_gates;
		dim_one_pos = dim_dense;
	}
	if(1) { //Make full connection on training result features
		dim_pi=0;
		dim_ps=extrafeat;
		old_dim_one_pos=0;
		//dim_ps remain;
		num_ps_gates=5;
		dim_dense=dim_ps+dim_pi;
		//  num_ps_gates=0; remains;
		num_pi_gates=dim_pi;
		num_gates=num_ps_gates+num_pi_gates;
		dim_one_pos = dim_dense;
		window_size=3;//atoi(params["WIN_SIZE_RET"].c_str());
	}
	num_states=num_label*(num_label+1); //number of  DUMMY.label plus number of label.label
	LocalWeights0 = (1 + num_states) * num_label;
	PsInterWeights0 = LocalWeights0 + num_states*num_gates;
	PiInterWeights0 = PsInterWeights0 + (window_size*(dim_ps+1)+1)*num_ps_gates;
	num_params = PiInterWeights0 + window_size*num_pi_gates*2 ;
	num_params += 1;
	dim_features = window_size*(dim_dense+1)+1;//To avoid incre
	cerr <<num_params<<","<<dim_ps<<","<<dim_pi<<","<<num_ps_gates<<","<<num_pi_gates<<","<<dim_features <<"," <<dim_one_pos<<","<<resultsize <<","<<dim_features<<endl;
	cerr <<"train data "<< trainData[0]->allresults.size()<<endl;
	cerr<< "test data "<<testData[0]->allresults.size()<<endl;
	cerr << "num_states: " << num_states << endl;
	cerr << "num_label: " << num_label << endl;
	cerr << "num_params: " << num_params << endl;
	cerr << "num_gates: " << num_gates << endl;
	cerr << "num_ps_gates: " << num_ps_gates << endl;
	cerr << "num_pi_gates: " << num_pi_gates << endl;
	cerr << "window_size: " << window_size << endl;
	cerr << "dim_features: " << dim_features << endl;
	cerr << "dim_one_pos: " << dim_one_pos << endl;
	cerr<<"Rebuild train data---------------\n";
	cerr<<"Extra feats dim "<<extrafeat<<"\n";
	cout<<"Rebuild train data---------------\n";
	//make new features
	for(int i=0; i<num_data; i++) {
		int df = dim_features*trainData[i]->length_seq;
		delete  trainData[i]->_features;
		trainData[i]->_features = new Score[df];
		for(int j=0; j<trainData[i]->length_seq; j++) {
			Score *tmp=trainData[i]->obs_feature[j];
			//      delete trainData[i]->obs_feature[j];
			trainData[i]->obs_feature[j]=new Score[dim_one_pos+1];
			//Preserve previous feature
			for(int k=0; k<old_dim_one_pos; k++) {
				trainData[i]->obs_feature[j][k]=tmp[k];
			}
			//Add the result as new features
			for(int k=0; k<resultsize; k++) {
				for(int s=0; s<num_label; s++) {
					trainData[i]->obs_feature[j][old_dim_one_pos+k*num_label+s]=exp(trainData[i]->allresults[k][j*num_label+s]);
				}
			}
			trainData[i]->obs_feature[j][dim_one_pos]=1;
			delete tmp;
		}
		trainData[i]->makeFeatures();
	}
	char foutname[500];
	sprintf(foutname,"%s-%d",trndata.c_str(),proc_id);
	retrndata=foutname;
	ofstream ofseq(retrndata.c_str());
	ofseq<<trainData.size()<<endl;
	for(int i=0; i<trainData.size(); i++) {
		ofseq<<(trainData[i]->length_seq)<<endl;
		for(int j=0; j<trainData[i]->length_seq; j++) {
			for(int k=0; k<dim_one_pos; k++) {
				ofseq<<trainData[i]->obs_feature[j][k]<<" ";
			}
			ofseq<<endl;
		}
		for(int j=0; j<trainData[i]->length_seq; j++) {
			ofseq<<trainData[i]->obs_label[j]<<endl;
		}
	}
	ofseq.close();
	cerr<<"output finished\n";
	cerr<<"Rebuild test data"<<old_dim_one_pos<<"\n";
	for(int i=0; i<num_tst; i++) {
		int df = dim_features*testData[i]->length_seq;
		delete  testData[i]->_features;
		testData[i]->_features = new Score[df];
		for(int j=0; j<testData[i]->length_seq; j++) {
			Score *tmp=testData[i]->obs_feature[j];
			//delete testData[i]->obs_feature[j];
			testData[i]->obs_feature[j]=new Score[dim_one_pos+1];
			for(int k=0; k<old_dim_one_pos; k++) {
				testData[i]->obs_feature[j][k]=tmp[k];
			}
			for(int k=0; k<resultsize; k++) {
				for(int s=0; s<num_label; s++) {
					testData[i]->obs_feature[j][old_dim_one_pos+k*num_label+s]=exp(testData[i]->allresults[k][j*num_label+s]);
				}
			}
			testData[i]->obs_feature[j][dim_one_pos]=1;
			delete tmp;
		}
		testData[i]->makeFeatures();
	}
	//  char foutname[500];
	sprintf(foutname,"%s-%d",tstdata.c_str(),proc_id);
	retstdata=foutname;
	ofseq.open(retstdata.c_str());
	ofseq<<testData.size()<<endl;
	for(int i=0; i<testData.size(); i++) {
		ofseq<<(testData[i]->length_seq)<<endl;
		for(int j=0; j<testData[i]->length_seq; j++) {
			for(int k=0; k<dim_one_pos; k++) {
				ofseq<<testData[i]->obs_feature[j][k]<<" ";
			}
			ofseq<<endl;
		}
		for(int j=0; j<testData[i]->length_seq; j++) {
			ofseq<<testData[i]->obs_label[j]<<endl;
		}
	}
	ofseq.close();
	cerr<<"output re test finished\n";
	/*
	  trainData.clear();
	  testData.clear();
	  LoadRetrainData();
	  for(int i=0;i<num_data;i++)
	  trainData[i]->makeFeatures();
	  for(int i=0;i<num_tst;i++)
	  testData[i]->makeFeatures();
	*/
	grad = new double[num_params];
	grad_sum = new double[num_params];
	weights = new double[num_params];
	if(!proc_id) {
		cerr <<"num_params = " << num_params << endl;
		for(int i=0; i<num_params; i++)
			weights[i]=(drand48()-drand48())/10;
	}
	for(int i=0; i<num_params; i++)
		reg[i] = 0;
	// for(int i=0;i<num_states*(1+num_states+num_gates);i++)
	//     reg[i] = 0;
#ifdef _MPI
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(weights, num_params, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(reg, num_params, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
#endif
}
int main(int argc, char **argv)
{
#ifdef _MPI
	MPI_Init(&argc, &argv);
#endif
	// the command line must be:
	//	bypass=1;
	extern char* optarg;
	// 	char c=0;
	// 	string model_dir="./", output_file="cnf_train.dat";
	// 	int w_size = 9; //best 13
	// 	int n_states = 101, n_gates = 20; //best 20
	// 	int n_local = 21;
	bCNF_Model cnfModel;
	bCNF_Model retrainModel;
	proc_id=0;
	num_procs=1;
#ifdef _MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
#endif
	if(proc_id==0)cerr<<"I am  "<< proc_id<<endl;
#ifdef _MPI
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
#endif
	cnfModel.Initialize(argc, argv);
	if(!proc_id) {
		cerr << "Initialization Finished!" << endl;
		cerr << "num_data = " << cnfModel.num_data*num_procs << endl;
		cerr << "num_tst = " << cnfModel.num_tst*num_procs << endl;
	}
	_LBFGS* lbfgs = new _LBFGS(&cnfModel);
	_LBFGS* lbfgsret = new _LBFGS(&retrainModel);
	if(cnfModel.params["ACT"] == "PREDICT") {
		cnfModel.Ensemble_start();
	} else if(cnfModel.params["ACT"] == "RETRAIN") {
		vector<double> w_init(cnfModel.num_params,0);
		for(int i=0; i<cnfModel.num_params; i++) {
			w_init[i]=cnfModel.weights[i];
		}
		//w_0 = w_init;
#ifdef _MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		if(cnfModel.params["RESUME"].length() > 0) {
			lbfgs->LBFGS(w_init,1);
		} else {
			lbfgs->LBFGS(w_init,500);
		}
#ifdef _MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		if(!proc_id) cerr<<"Begin retraining\n";
		cnfModel.Double_train();// rebuild all the feature needed.
		if(!proc_id)cerr<<"retrain feature rebuilt\n";
#ifdef _MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		//vector<double> w_init(cnfModel.num_params,0);
		vector<double> w_initret(cnfModel.num_params,0);
		for(int i=0; i<cnfModel.num_params; i++) {
			w_initret[i]=cnfModel.weights[i];
		}
		lbfgs = new _LBFGS(&cnfModel);
		if(!proc_id)cerr<<"begin lbfgs\n";
#ifdef _MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		lbfgs->LBFGS(w_initret,500);
#ifdef _MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
	} else {
		vector<double> w_init(cnfModel.num_params,0);
		for(int i=0; i<cnfModel.num_params; i++) {
			w_init[i]=cnfModel.weights[i];
		}
		//w_0 = w_init;
#ifdef _MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		if(!proc_id)cerr<<"[TRAIN]begin lbfgs\n";
		if(cnfModel.params["PREDICT1"].length() > 0) {
			lbfgs->Report(w_init,1,0,0);
		} else {
			lbfgs->LBFGS(w_init,1500);
		}
#ifdef _MPI
		MPI_Barrier(MPI_COMM_WORLD);
#endif
	}
	for(int i=0; i<cnfModel.num_data; i++)
		delete cnfModel.trainData[i];
	for(int i=0; i<cnfModel.num_tst; i++)
		delete cnfModel.testData[i];
#ifdef _MPI
	MPI_Finalize();
#endif
	return 0;
}
