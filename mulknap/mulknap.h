
/* ======================================================================
	     mulknap.h,  David Pisinger                     feb 1998 
   ====================================================================== */

/* header file for the MULKNAP algorithm */


/* ======================================================================
                                debug variables
   ====================================================================== */

extern long iterates;
extern long tightened;
extern long reduced;
extern long gub;
extern long tottime;
extern long coresize;


/* ======================================================================
                                mulknap
   ====================================================================== */

extern int mulknap(int n, int m, int *p, int *w, int *x, int *c);


