#include<iostream>
using namespace std;
int main()
{
    int n,m;
    cout<<"Enter the number till where you want the prime numbers ";
    cin>>n;
    if(n<2)
    {
         cout<<"Sorry no prime numbers exist before "<<n;
    }
    for(int i=2;i<=n;i++)
    {
        m = 0;
        for(int j=2;j<i;j++)
        {
            if(i%j==0)
            {
                m+=1;
                break;
            }
        }
        if(m==0)
        {cout<<i<<endl;}
    }
    return 0;
}
