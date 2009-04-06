LEX = flex
YACC = yacc

all: param_parser.cpp param_lexer.c

param_parser.cpp: params.y
	$(YACC) -o $@ -d params.y

param_lexer.c: params.l
	$(LEX) -o $@ params.l
