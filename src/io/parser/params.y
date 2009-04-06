%{
#include <stdio.h>
#include <string.h>
#include "../PVParams.hpp"

PV::PVParams* handler;

#ifdef __cplusplus
extern "C" {
#endif

   int yyparse(void);
   int yylex(void);

   void yyerror(const char * str)
   {
      fprintf(stderr,"error: %s\n", str);
   }
 
   int yywrap()
   {
      return 1;
   } 
  
#ifdef __cplusplus
}
#endif

int pv_parseParameters(PV::PVParams* action_handler)
{
   handler = action_handler;
   return yyparse();
}

%}

%union {char * sval; double dval; }
%token <sval> T_STRING
%token <sval> T_KEYWORD
%token <sval> T_ID
%token <dval> T_NUMBER
%type  <sval> name

%%

parameter_groups : /* empty */
                 | parameter_groups parameter_group
                 ;

parameter_group : T_KEYWORD name '='
                   '{'
                        parameter_defs
                   '}' ';'
        { handler->action_parameter_group($1, $2); }
              ;

name : T_STRING
        { $$ = $1; }
     ;

parameter_defs : /* empty */
               | parameter_defs parameter_def
               ;

parameter_def : T_ID '=' T_NUMBER ';'
                 { handler->action_parameter_def($1, $3); }
              ;
