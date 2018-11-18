#include <iostream>
#include <string>
#include<stdio.h>
#include<cmath>
#include <sstream>
#include <iostream>
#include <vector>

using namespace std;
// enum for cases
enum ACTION_TYPE {ENCRYPT = 1, DECRYPT= 2};

// thanks to cs50 by D Malan at Harvard for the inspiration
class crypt {
protected:
    char str[100]; // maybe string?
public:
    // set text for all subclasses
    void setText(char *strin){
        strcpy(str,strin);
    }
    
    virtual void encrypt() = 0;
    virtual void decrypt() = 0;
    
    void action(int y){
        switch(y){
            case ENCRYPT: encrypt();
                break;
            case DECRYPT: decrypt();
                break;
        }
    }
    
};

class caeser: public crypt {
protected:
    int i;
  public:
    
    //encryption with a keyvalue of 2
    void encrypt() {
        for(i=0;(i<100 && str[i]!='\0');i++){
            str[i] = (char)((abs((int)str[i] - 97 + 2))%26 + 97);
        }
        cout << str << endl;
    }
    
    void decrypt() {
        for(i=0;(i<100 && str[i]!='\0');i++){
            str[i] = (char)((abs((int)str[i] -97- 2))%26 + 97);
        }
        cout << str << endl;
    }
    
};



class vigenere: public crypt {
protected:
    int i,j;
    char key[8];
    int keyLen;
    int msgLen;
    char newKey[100];
    
    
public:
    
    void setKey(char *strin){
        strcpy(key,strin);
        keyLen = strlen(key);
        msgLen = strlen(str);
        for(i=0,j=0;i<msgLen;++i,++j){
            if(j==keyLen)
                j = 0;
            newKey[i] = key[j];
        }
        newKey[i] = '\0';
    }
    
    
    void encrypt() {
        for(i=0;i<msgLen;i++){
            str[i] = (char)((((int)str[i] - 97) + ((int)newKey[i]-97))%26 + 97);
        }
        cout << str << endl;
    }
    void decrypt() {
        for(i=0;i < msgLen;i++){
            str[i] = (char)(( (abs((int)str[i] -97) - ((int)newKey[i]-97))%26) + 97);
        }
        cout << str << endl;
    }
    
};

//weblink and reference for b64 algo https://github.com/philipperemy/easy-encryption
class b64: public crypt {
public:
    
    void encrypt() {
        
        std::string out;
        
        int val=0, valb=-6;
        for (int jj = 0; jj < strlen(str); jj++) {
            char c = str[jj];
            val = (val<<8) + c;
            valb += 8;
            while (valb>=0) {
                out.push_back("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[(val>>valb)&0x3F]);
                valb-=6;
            }
        }
        if (valb>-6) out.push_back("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[((val<<8)>>(valb+8))&0x3F]);
        while (out.size()%4) out.push_back('=');
        cout << out << endl;
    }
    
    void decrypt() {
        std::string out;
        
        std::vector<int> T(256,-1);
        for (int i=0; i<64; i++) T["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] = i;
        
        int val=0, valb=-8;
        for (int jj = 0; jj < strlen(str); jj++) {
            char c = str[jj];
            if (T[c] == -1) break;
            val = (val<<6) + T[c];
            valb += 6;
            if (valb>=0) {
                out.push_back(char((val>>valb)&0xFF));
                valb-=8;
            }
        }
        cout << out << endl;
    }
};





// tester class to iterate over algos and apply action
/*
class Tester {
    
    std::vector<crypt> algos;
    
    void addCryptAlgorithm(crypt a) {
        algos.push_back(a);
    }
    
    void act(ACTION_TYPE action) {
        
        for (auto algo: algos) {
            algo.setText()
            algo.action(action)
        }
    }
}
*/

int main () {
    int i,x;
    char strin[100];
    // for vigenere cipher/ can be changed to anything
    char keyval[] = "newyork";
    cout << "please enter a string:\t";
    cin >> strin;
    
    cout << "\n Please choose following options:\n";
    cout << "1 = encrypt\n";
    cout << "2 = decrypt\n";
    cin >> x;
    caeser algo;
    algo.setText(strin);
    cout << "caesar cipher:\n";
    algo.action(x);
    
    vigenere vig;
    vig.setText(strin);
    vig.setKey(keyval);
    cout << "vigenere cipher:\n";
    vig.action(x);
    
    b64 bshift64;
    bshift64.setText(strin);
    cout << "b64 cipher:\n";
    bshift64.action(x);
    
    
  return 0;
}
