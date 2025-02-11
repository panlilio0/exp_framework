
#ifndef SPIKYNET_H
#define SPIKYNET_H
#include <ode/ode.h>
#include <vector.h>
#include "myutils.h"
#include <queue.h>
//#include "Network.h"

#define SPIKE_DECAY 0.1
#define MAX_BIAS 10

class SpikyNode
{
 
 protected:
  vector<float> _weights;               //input weights:30
  float level;
  void init(int size);
  vector<int> firelog;

 public:
  SpikyNode(int size);
  ~SpikyNode();
  float bias();
  void set_bias(float);
  void print();
  void fprint(FILE *);
  float compute(vector<float> inputs);
  void setWeights(vector<float> inws);
  void set_weight(int,float);
  float dutyCycle();
};


class SpikyLayer
{
 protected:
  vector<SpikyNode *> _nodes;
  void init(int numnodes, int numinputs);
 public:
  SpikyLayer(int numnodes, int numinputs);
  ~SpikyLayer();
  void fprint(FILE *);
  void print();
  vector<float> compute(vector<float> inputs);
  void setWeights(vector<float> inws);
  vector<float> dutyCycles();
};

class SpikyNet
{
 protected:

  SpikyLayer *_hiddenLayer;
  SpikyLayer *_outputLayer;

  virtual void init(int,int,int);
  

 public:
  SpikyNet(int,int,int);
  ~SpikyNet();
  vector<float> compute(vector<float> input);
    void fprint(char *);
  void setWeights(vector<float> inws);
  void print();
};
#endif
