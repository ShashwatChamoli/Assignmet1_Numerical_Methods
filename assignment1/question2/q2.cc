#include<iostream>
#include<limits>
using namespace std;
int main()
{
cout<<"This part of the code shows the size of some data types"<<endl;
cout<<"Size of int : "<<sizeof(int)<<endl;
cout<<"Size of float : "<<sizeof(float)<<endl;
cout<<"Size of bool : "<<sizeof(bool)<<endl;
cout<<"Size of char : "<<sizeof(char)<<endl;
cout<<"This part of the code shows the maximum and minimum values a data type can take "<<endl;
cout<<"The maximum value of int is ";
std::cout<<std::numeric_limits<int>::max();
cout<<"\n";
cout<<"The minimum value of int is ";
std::cout<<std::numeric_limits<int>::min();
cout<<"\n";
cout<<"The maximum value of float is ";
std::cout<<std::numeric_limits<float>::max();
cout<<"\n";
cout<<"The minimum value of float is ";
std::cout<<std::numeric_limits<float>::min();
cout<<"\n";
cout<<"The maximum value of long int is ";
std::cout<<std::numeric_limits<long int>::max();
cout<<"\n";
cout<<"The minimum value of long int is ";
std::cout<<std::numeric_limits<long int>::min();
return 0;
}
