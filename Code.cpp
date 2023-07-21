#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <cstdlib>
#include <time.h>
using namespace std;

//1. Class Country_Stats to store the data about the countries
class Country_Stats
{
  public:

  string country; //Name of the country
  vector<double> stats; //stores the 9 attributes
  void print_country()
  {
      cout<< country <<" ";
      for(auto x:stats) cout<< x<<" ";
      cout<<endl;
  }
};

//2. Neurons are treated as vectors of number of dimensions equal to the number of features in the input data
class Neuron
{
    public:

    int weight_size; //equal to number of features
    vector<double>weights; //weights vector
    pair<int,int>coordinate; //location of neuron in lattice

    void print_neuron_weights()
    {
        for(auto x: weights) cout<<x<<" ";
        cout<<endl;
    }

    void set_neuron_coordinates(const int weight_size,int x_cord,int y_cord)
    {
        this->weight_size=weight_size;
        this->weights.resize(weight_size,0.0);
        this->coordinate.first=x_cord;
        this->coordinate.second=y_cord;
        return;
    }
    //Randomly Initialize neuron weights
    void initialize_neuron_random()
    {
        srand((unsigned)time(NULL));
        for (int i=0;i<this->weight_size;i++)
        {
            double random_number_oto1= (double)(rand()/(1.0*RAND_MAX));
            //a random number from 0 to 1 is assigned to weights
            weights[i]=random_number_oto1;
        }
        return;
    }
};

//3. A Lattice class to organize the neurons. The SOM algorithm will operate on this
class Lattice
{
  public:

    int lattice_length;
    int lattice_width;
    vector<vector<Neuron>> lattice;
    //Constructors
    Lattice(){}
    Lattice(int length, int width)
    {   //set size of lattice if given
        lattice_length=length;
        lattice_width=width;
        lattice.resize(length,vector<Neuron>(width));
    }

    //Define positions of neurons and initialize their weights
    void initialize_Lattice_Neurons(const int number_of_features)
   {
       for(int i=0;i<lattice_length;i++)
        for(int j=0;j<lattice_width;j++)
        {
          lattice[i][j].set_neuron_coordinates(number_of_features,i,j);//Set position and size of neurons
          lattice[i][j].initialize_neuron_random();//Initialize weights
        }
   }
};

//4. Contains all the Functions used to implement the SOM Algorithm
class Self_Organising_Map
{
  public:
  double learning_rate;//Name is self explanatory
  int number_of_epochs;//The number of Iterations for which the Algorithm will run

  //Parameterized Constructor
  Self_Organising_Map(double lr, int num_epochs)
  {
    learning_rate=lr;
    number_of_epochs=num_epochs;
  }

  //a. Find Distance between a Neuron's weight vector and a Row vector of Data
  double find_neuron_distance(const Neuron& neuron, const Country_Stats &country_stats)
  {
    double ans=0.0;
    int number_of_features= neuron.weight_size;
    for(int i=0;i<number_of_features;i++)
      ans+= (country_stats.stats[i] -neuron.weights[i])*(country_stats.stats[i] -neuron.weights[i]);
    //Return the Euclidean distance
    return (double) sqrt(ans);
  }

  //b. Find the Winner Neuron for a given country
  Neuron find_winner_neuron(const Lattice& SOM, const Country_Stats &country_stats)
  {
    //Initialize winner neuron with the first neuron of lattice
    Neuron winner=SOM.lattice[0][0];
    double min_distance=find_neuron_distance(winner,country_stats);
    //For all neurons of the Lattice
    for(int i=0;i<SOM.lattice_length;i++)
      for(int j=0;j<SOM.lattice_width;j++)
      {
        double curr_neuron_distance=find_neuron_distance(SOM.lattice[i][j],country_stats);
        //If distance of current neuron is less than the winner till now
        if(curr_neuron_distance< min_distance)
        {
          winner=SOM.lattice[i][j];//Declare current neuron as winner
          min_distance=curr_neuron_distance;//Update the min distance with new winner disatnce
        }
      }
    return winner;
  }

  //c. Update the neurons' weights based on its distance(neighborhood) with the winner.
  // The more closer a neuron is to the winner, more its weights are affected
  void update_neuron_weights(Lattice& SOM, Neuron &winner, const Country_Stats &country_stats)
  {
    int winnerx=winner.coordinate.first;
    int winnery=winner.coordinate.second;
    int num_features=winner.weight_size;
    //For all neurons of the lattice
    for(int neuronx=0;neuronx < SOM.lattice_length;neuronx++)
        for(int neurony=0;neurony < SOM.lattice_width;neurony++)
        {
            //Calculate the distance between the neuron and the winner and the neighborhood function
            int d=(neuronx-winnerx)*(neuronx-winnerx) + (neurony-winnery)*(neurony-winnery);
            double neighbourhood_function=exp(-pow(d,0.5));
            //Update the neuron's weight vector
            for(int i=0;i<num_features;i++)
            {
                // wij = wij(old) + alpha(t) * neighbourhood_finction * (xik - wij(old))
                SOM.lattice[neuronx][neurony].weights[i]+= learning_rate*neighbourhood_function*(country_stats.stats[i]- SOM.lattice[neuronx][neurony].weights[i]);
            }
        }
  }

  //d. Change the learning rate based on the number of iterations done till now and the total number of epochs
  double change_learning_rate(const int epochs)
  {
    return (double)((learning_rate*(epochs+1))/(number_of_epochs*1.0));
  }

  //e. Integrate all the functionalities and train the Self Organizing Map
  void simulate_SOM(Lattice& SOM_lattice, const vector<Country_Stats> & country_data, const int number_of_features)
  {
    //Initialize the Lattice of neurons
    SOM_lattice.initialize_Lattice_Neurons(number_of_features);

    //Repeat for number of epochs (iterations)
    for(int epoch=0;epoch< number_of_epochs;epoch++)
    {
      // for each country
      SOM_lattice.lattice[1][1].print_neuron_weights();
      for(int i=0;i<country_data.size();i++)
      {
        //Find the winner and update the weights of neurons
        Neuron winner= find_winner_neuron(SOM_lattice,country_data[i]);
        update_neuron_weights(SOM_lattice,winner,country_data[i]);
      }
      //Change the learning rate after each epoch
      learning_rate=change_learning_rate(epoch);
    }
  }
};


int main()
{
    vector<Country_Stats> country_data;
     int number_of_features=9;
    // **Reading the data**

    fstream fin;
    fin.open("C:\\Users\\kaila\\Downloads\\Country-data-refined.csv", ios::in);
    //Temporary variables to help in reading the file
    vector<string> row;
    string line, word;
    Country_Stats temp_country;
    //Consume the first row of headers
    getline(fin, line);

    while (getline(fin, line))
    {   //clear the row variable and store new row of data in it
        row.clear();
        stringstream s(line);
        while (getline(s, word, ','))
          row.push_back(word);
        //store data in temp_country from row variable
        temp_country.country= row[0];
        temp_country.stats.clear();
        for(int i=1;i<row.size();i++) temp_country.stats.push_back(stod(row[i]));
        //push this temp_country in country_data
        country_data.push_back(temp_country);
    }

    //**Action Time**

   int proposed_lattice_length=3;
   int proposed_lattice_width=3;
   //Instantiating
   Lattice SOM_lattice(proposed_lattice_length,proposed_lattice_width);
   Self_Organising_Map SOM(0.5,10);
   //Train the Self Organizing Map
   SOM.simulate_SOM(SOM_lattice,country_data,number_of_features);

   //Results
   //unordered_map<pair<int,int>,vector<string>> clusters;

   for(int i=0;i<10;i++)
   {
       Neuron temp = SOM.find_winner_neuron(SOM_lattice,country_data[5*i+3]);
       cout<<country_data[10*i].country<< " "<<temp.coordinate.first <<" "<<temp.coordinate.second<<endl;
   }
    return 0;
}
