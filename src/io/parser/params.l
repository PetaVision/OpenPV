%{
#include <stdio.h>
#include <float.h>
#include "param_parser.hpp"
%}
%option nounput
%option noinput
%option noyyget_extra
%option noyyset_extra

%%
([+]|[-])?(([0-9]+(\.[0-9]*)?)|([0-9]*\.[0-9]+)|([0-9](\.[0-9]*)?[Ee]([+]|[-])?[0-9]+)) {
   yylval.dval = strtod(yytext, NULL);
   return T_NUMBER;
}

[+]?(infinity) { yylval.dval = FLT_MAX; return T_NUMBER; }
-infinity { yylval.dval = -FLT_MAX; return T_NUMBER; }

"true" { yylval.dval = 1; return T_NUMBER; }
"false" { yylval.dval = 0; return T_NUMBER; }

[a-zA-Z][a-zA-Z0-9_]* {
   yylval.sval = strdup(yytext);
   return T_ID;
}

[\"][a-zA-Z0-9_ ]*[\"] { yylval.sval = strdup(yytext); return T_STRING; }
[\"][a-zA-Z0-9_~+/.-]+[\"] { yylval.sval = strdup(yytext); return T_FILENAME; }

"{"        { return *yytext; }
"}"        { return *yytext; }
"="        { return *yytext; }
";"        { return *yytext; }
":"        { return *yytext; }

"#include" { return T_INCLUDE; }

[//].*   /* ignore C++ style comments */

[ \t\n]+ /* ignore whitespace */

%%
