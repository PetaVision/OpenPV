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

"#include" { return T_INCLUDE; }
"ParameterSweep" {return T_PARAM_SWEEP; }
"BatchSweep" {return T_BATCH_SWEEP; }
"@"[a-zA-Z][a-zA-Z0-9_]* {
    yylval.sval = strdup(yytext);
    return T_ID_OVERWRITE;
}

([+]|[-])?(([0-9]+(\.[0-9]*)?)|([0-9]*\.[0-9]+)|([0-9](\.[0-9]*)?[Ee]([+]|[-])?[0-9]+)) {
   yylval.dval = strtod(yytext, NULL);
   return T_NUMBER;
}

[+]?(infinity) { yylval.dval = FLT_MAX; return T_NUMBER; }
-infinity { yylval.dval = -FLT_MAX; return T_NUMBER; }

"true" { yylval.dval = 1; return T_NUMBER; }
"false" { yylval.dval = 0; return T_NUMBER; }

"NULL" { yylval.sval = NULL; return T_STRING; }

[a-zA-Z][a-zA-Z0-9_]* {
   yylval.sval = strdup(yytext);
   return T_ID;
}

[\"][a-zA-Z0-9_ ]*[\"] { yylval.sval = strdup(yytext); return T_STRING; }
[\"][a-zA-Z0-9_~+/.-]+[\"] { yylval.sval = strdup(yytext); return T_FILENAME; }

"{"        { return *yytext; }
"}"        { return *yytext; }
"["        { return *yytext; }
"]"        { return *yytext; }
"="        { return *yytext; }
","        { return *yytext; }
";"        { return *yytext; }
":"        { return *yytext; }


[//].*   /* ignore C++ style comments */

[ \t\n]+ /* ignore whitespace */

%%
