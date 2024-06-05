#include <stdio.h>
using namespace std;
struct ListNode
{
    int value;
    ListNode* next;
};

ListNode* createLinkedList () {
    ListNode* head = nullptr;
    ListNode* current = nullptr;

    for (int i = 1; i <= 5; ++i) {
        ListNode* newNode = new ListNode;
        newNode->value = i;
        newNode->next =nullptr;

        if (head == nullptr) {
            head = newNode;
            current = head;
        } else {
            current->next = newNode;
            current = newNode;
        }
    }
    return head;
}




int main(){
    return 0;
}



 