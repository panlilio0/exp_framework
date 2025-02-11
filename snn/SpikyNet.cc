
#include "SpikyNet.h"

SpikyNode::SpikyNode(int size)
{
  init(size);
}

SpikyNode::~SpikyNode()
{
}

void SpikyNode::init(int size)
{
  // printf("spikynode init %d\n",size);
  firelog.clear();
  if (size > 0)
    {
      _weights.clear();
      for (int i = 0; i < size; i++)
	{
	  
	  _weights.push_back((dRandReal()*2.0)-1);
	  
	}
      
      //and now the bias
      
      _weights.push_back(dRandReal()*MAX_BIAS);
    }
  
}

void SpikyNode::fprint(FILE *ofile)
{
  fprint_vector_of_floats(ofile,_weights);
}

float SpikyNode::compute(vector<float> inputs)
{

  while (firelog.size() > 200)
    firelog.erase(firelog.begin());//pop from front

  level -= SPIKE_DECAY;

  if ((inputs.size()+1) != _weights.size())
    {
      printf("spiky::compute error: %d inputs, %d weights (+1 bias)\n",inputs.size(),_weights.size()-1);
      return 0.;
    }
 
  float sum = 0.;
  
  for (uint i = 0; i < inputs.size(); i++)
    {
      sum+= inputs[i]*_weights[i];
    }
  
  level += sum;

  // printf("SpikyNode::compute - sum is %f, level is %f, bias is %f\n",sum,level,bias());
  level = (level > 0) ? level : 0;

  if (level >= bias())
    {
      //      printf("fire!\n");
      level = 0;
      firelog.push_back(1);
      return 1;
    }
  firelog.push_back(0);
  return 0;

    

  //if (dutyCycle() > .30)
  //  return 1;
  // don't normalize!
  //sum = sum/(_weights.size()*1.0); //add one to account for bias

  //  return sigmoid(sum);
}

float SpikyNode::dutyCycle()
{
  int fires = 0;
  for (uint i = 0; i < firelog.size(); i++)
    {
      //      printf("%d",firelog[i]);
      if (firelog[i] == 1)
	fires++;
    }

  //printf("\n");

  if (firelog.size() > 30)
    return fires*1.0/firelog.size();
  else
    return 0;

}
void SpikyNode::setWeights(vector<float> inws)
{
  if (inws.size() != _weights.size())
    printf("size mismatch in NNode:setweights\n");

  
  _weights.clear();
  for (uint i = 0; i < inws.size(); i++)
    {
      _weights.push_back(inws[i]);
    }

}

float SpikyNode::bias()
{
  return _weights[_weights.size()-1];
}

void SpikyNode::set_weight(int i,float val)
{

  if ((i < 0) || (i >= (int)_weights.size()))
    printf("NNode::set_weight bad value %d!\n",i);

  _weights[i] = val;
  
}

void SpikyNode::set_bias(float val)
{
  _weights[_weights.size()-1] = val;
}

void SpikyNode::print()
{
  print_vector_of_floats(_weights);
}

/**********************************************************************************/

SpikyLayer::SpikyLayer(int numnodes, int numinputs)
{
  init(numnodes,numinputs);
}

SpikyLayer::~SpikyLayer()
{
   for (uint i = 0; i < _nodes.size(); i++)
    {
      delete(_nodes[i]);
      _nodes[i] = NULL;
    }
  _nodes.resize(0);
  _nodes.clear();
}

void SpikyLayer::init(int numnodes, int numinputs)
{
  //  printf("spikylayer init %d %d\n",numnodes,numinputs);
  _nodes.clear();
  for (int i = 0; i < numnodes; i++)
    {
      _nodes.push_back(new SpikyNode(numinputs));
    }

  // print();
}

void SpikyLayer::fprint(FILE *ofile)
{
  for (uint i = 0; i < _nodes.size(); i++)
    {
      _nodes[i]->fprint(ofile);
    }
}
void SpikyLayer::print()
{
  for (uint i = 0; i < _nodes.size(); i++)
    {
      printf("node %d:",i);
      _nodes[i]->print();
    }
}

vector<float> SpikyLayer::compute(vector<float> inputs)
{
 
  vector<float> results;
  results.clear();
  for (uint i = 0; i < _nodes.size(); i++)
    {
      results.push_back(_nodes[i]->compute(inputs));
    }
  
  return results;
}
void SpikyLayer::setWeights(vector<float> inws)
{
  int weightspernode = (int)(inws.size()/_nodes.size());

  vector<float> curweights;
  curweights.clear();

  //  printf("%d weights per node\n",weightspernode);
  for (uint i = 0; i < _nodes.size(); i++)
    {
      curweights.clear();
      for (int j = 0; j < weightspernode; j++)
	{
	  curweights.push_back(inws[weightspernode*i + j]);
	}
      _nodes[i]->setWeights(curweights);
    }
}

vector<float> SpikyLayer::dutyCycles()
{
  vector<float> results;
  results.clear();
  for (uint i = 0; i < _nodes.size(); i++)
    {
      results.push_back(_nodes[i]->dutyCycle());
    }
  return results;

}
/**********************************************************************************/

SpikyNet::SpikyNet(int inputs, int hidden, int outputs)
{
  init(inputs,hidden,outputs);
}

SpikyNet::~SpikyNet()
{
  delete(_hiddenLayer);
  delete(_outputLayer);
}

void SpikyNet::print()
{
  //  getchar();
  //  printf("hello\n");
  printf("Hidden Layer:\n");
  _hiddenLayer->print();
  printf("Output Layer:\n");
  _outputLayer->print();
}

void SpikyNet::fprint(char *fname)
{
  FILE *ofile = fopen(fname,"w+");
  _hiddenLayer->fprint(ofile);
  _outputLayer->fprint(ofile);
  fclose(ofile);
}

void SpikyNet::init(int inputs, int hidden, int outputs)
{
  //printf("spiky init\n");
  _hiddenLayer = new SpikyLayer(hidden,inputs);
  _outputLayer= new SpikyLayer(outputs,hidden);
}  
  
vector<float> SpikyNet::compute(vector<float> input)
{

  //  printf("spiky compute\n");
  vector<float> HLvals = _hiddenLayer->compute(input);
  //printf(" * ");
  //  print_vector_of_floats(HLvals);
  vector<float> OLvals = _outputLayer->compute(HLvals);

  vector<float> OLrates = _outputLayer->dutyCycles();
  //  print_vector_of_floats(OLvals);
  return OLrates;
  //  return OLvals;

  //get inputs
  //send inputs thru hidden
  //send hidden thru out
  //push back out
}

void SpikyNet::setWeights(vector<float> inws)
{
  vector<float> hws;
  hws.clear();
  vector<float> ows;
  ows.clear();
 
  int insize = (int)inws.size();
  for (uint i = 0; (int)i < (insize/2); i++)
    {
      hws.push_back(inws[i]);
      ows.push_back(inws[i+insize/2]);
    }

  _hiddenLayer->setWeights(hws);
  _outputLayer->setWeights(ows);
}
