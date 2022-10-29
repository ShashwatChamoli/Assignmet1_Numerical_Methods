#include <iostream>
using namespace std;
int random(n)
{
    
}
int main()
{
srand((unsigned) time(0));
int n,c;
cout<<"Plaese enter the size of the array ";
cin>>n;
int a[n],b[20];
for (int i=0;i<n;i++)
{
a[i]=rand();
}
cout<<"Random numbers are "<<endl;
for(int i=0;i<n;i++)
{
cout<<a[i]<<" ";
}
cout<<"\n20 random numbers are "<<endl;
for(int i=0;i<20;i++)
{
b[i]=rand();
cout<<b[i]<<" ";
}
cout<<"\n20 random numbers in ascending order are "<<endl;
for(int j=0;j<20;j++)
{
for(int i=0;i+1<20-j;i++)
{
if(b[i]>b[i+1])
{
c=b[i];
b[i]=b[i+1];
b[i+1]=c;
}
}
}
for(int i=0;i<20;i++)
{
cout<<b[i]<<" ";
}
}
