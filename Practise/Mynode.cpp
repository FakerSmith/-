class MyLinkedList {
    
struct Mynode {
    int val;
    Mynode* next;
     Mynode(int x) : val(x),next(nullptr) {}
};
    
Mynode* head;
int size;
public:
    MyLinkedList() {
        head = new Mynode(0);
        size = 0;
    }
    
    int get(int index) {
        if (index >= size || index < 0) {
            return -1;
        }
        
        Mynode* cur = head;
        for (int i=0; i<=index; ++i) {
            cur = cur->next;
        }
        return cur->val;
    }
    
    void addAtHead(int val) {
        Mynode* cur = new Mynode(val);
        cur->next = head->next;
        head->next = cur;
        size++;
    }
    
    void addAtTail(int val) {
        Mynode* cur =new Mynode(val);
        Mynode* point = head;
        for (int i= 0; i< size; ++i){
            point = point->next;
        }
        point->next = cur;
        size++;
    }
    
    void addAtIndex(int index, int val) {
        if (index <= 0)
        {
            addAtHead(val);
            return;
        }

        else if (index == size)
        {
            addAtTail(val);
            return;
        }

        else if (index < size && index > 0)
        {
            Mynode* temp = head;
            for (int i = 0; i < index; i++)
            {
                temp = temp->next;
            }

            Mynode* New_Node = new Mynode(val);
            New_Node->next = temp->next;
            temp->next = New_Node;

            size++;
        }
    }

 
    
    void deleteAtIndex(int index) {
        Mynode* temp = head;

        if (index < size && index >=0 && size>=1)
        {
            for (int i = 0; i < index ; i++)
            {
                temp = temp->next;
            }

            Mynode* temp_delete = temp->next;
            temp->next = temp->next->next;
            delete temp_delete;

            size--;
        }
    }
};
//链表的几种功能