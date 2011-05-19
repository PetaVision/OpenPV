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
%token <sval> T_ID
%token <dval> T_NUMBER
%token <sval> T_FILE_KEYWORD
%token <sval> T_FILENAME
%type  <sval> parameter_group

%%

declarations : /* empty */
             | declarations parameter_group
             | declarations filename_def
             ;

parameter_group : T_ID T_STRING '=' '{' parameter_defs '}' ';'
                      { handler->action_parameter_group($1, $2); }
                  ;
                
parameter_defs : /* empty */
               | parameter_defs parameter_def
               | parameter_defs parameter_string_def
               ;

parameter_def : T_ID '=' T_NUMBER ';'
                 { handler->action_parameter_def($1, $3); }
              ;

parameter_string_def : T_ID '=' T_STRING ';'
                        { handler->action_parameter_string_def($1,$3); }
                     | T_ID '=' T_FILENAME ';'
                        { handler->action_parameter_string_def($1,$3); }
                     ;

filename_def : T_FILE_KEYWORD T_STRING '=' T_FILENAME ';'
                { handler->action_filename_def($2, $4); }
             | T_FILE_KEYWORD T_STRING '=' T_STRING ';'
                { handler->action_filename_def($2, $4); }
             ;
