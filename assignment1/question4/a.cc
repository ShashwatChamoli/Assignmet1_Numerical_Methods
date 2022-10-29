#include<iostream>
using namespace std;
int random(int n)
{
    int a[n];
    for (int i=0;i<n;i++)
    {
        a[i]=rand();
        cout<<n;
    }
 return 0;
}
int main()
{
    int n;
    srand((unsigned) time(0));
    cout<<"Enter the size of the array ";
    cin>>n;
    cout<<n<<" random numbers are ";
    cout<<random(n);
    cout<<"\n20 random numbers are "<<endl;
    cout<<random(20);
    cout<<"\n20 random numbers in ascending order are "<<endl;

}
