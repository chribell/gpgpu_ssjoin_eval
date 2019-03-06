#ifndef BITMAP_GPUSSJOIN_HOST_INPUT_HXX
#define BITMAP_GPUSSJOIN_HOST_INPUT_HXX

#include <cstring>
#include "classes.hxx"

#define LINEBUFSIZE 8192

struct tokenize_whitespace {
    const char * p;
    const char * b;
    bool isend;
    char curtoken[LINEBUFSIZE];

    inline tokenize_whitespace() : p(NULL), b(NULL), isend(true) {}

    void setline(const char * line) {
        p = line;
        while(*p == ' ' || *p == '\t' || *p == '\n') {
            p += 1;
        }
        b = p;
        isend = *p == 0;
    }

    char inline * next() {
        bool copied = false;
        while(true) {
            if(*p == ' ' || *p == '\t' || *p == 0 || *p == '\n') {
                if(p != b) {
                    size_t len = std::min<size_t>(LINEBUFSIZE - 1, p-b);
                    memcpy(curtoken, b, len);
                    *(curtoken + len) = 0;
                    copied = true;
                }
                if(*p == 0) {
                    isend = true;
                    break;
                }

                p  += 1;
                b = p;
                continue;

            } else if(copied) {
                break;
            }

            ++p;
        }
        return curtoken;
    }

    inline bool end() { return isend; }
};
void add_input(const std::string & filename, Collection<unsigned int>& collection);

#endif //BITMAP_GPUSSJOIN_HOST_INPUT_HXX
