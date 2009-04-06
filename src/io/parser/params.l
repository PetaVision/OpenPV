%{
#include <stdio.h>
#include "param_parser.hpp"
%}

%%

"HyPerCol"     { yylval.sval = strdup(yytext); return T_KEYWORD; }
"HyPerConn"    { yylval.sval = strdup(yytext); return T_KEYWORD; }

"PVLayer"      { yylval.sval = strdup(yytext); return T_KEYWORD; }
"HyPerLayer"   { yylval.sval = strdup(yytext); return T_KEYWORD; }
"V1"           { yylval.sval = strdup(yytext); return T_KEYWORD; }

[a-zA-Z][a-zA-Z0-9_]* {
   yylval.sval = strdup(yytext);
   return T_ID;
}

([+]|[-])?(([0-9]+(\.[0-9]*)?)|([0-9]*\.[0-9]+)) {
   yylval.dval = atof(yytext);
   return T_NUMBER;
}

[\"][a-zA-Z0-9_ ]*[\"] { yylval.sval = strdup(yytext); return T_STRING; }

"{"        { return *yytext; }
"}"        { return *yytext; }
"="        { return *yytext; }
";"        { return *yytext; }

[//].*   /* ignore C++ style comments */

[ \t\n]+ /* ignore whitespace */

%%
