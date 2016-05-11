%{
#include <stdio.h>
#include <string.h>
#include <io/PVParams.hpp>

PV::PVParams* handler;

/* In MPI, root process reads the file and broadcasts it to all processes, and
 * each process parses the file from memory independently.  The problem is that
 * by default, yacc/bison parsers read from standard input.  As a workaround
 * start block of stuff copied from param_lexer.c, which is created by params.l
 * There ought to be an easier way to do this; it's (YY|yy)* variables and this
 * is a .y file
 */
#ifndef YY_TYPEDEF_YY_SIZE_T
#define YY_TYPEDEF_YY_SIZE_T
typedef size_t yy_size_t;
#endif

#ifdef __cplusplus

/* The "const" storage-class-modifier is valid. */
#define YY_USE_CONST

#else	/* ! __cplusplus */

/* C99 requires __STDC__ to be defined as 1. */
#if defined (__STDC__)

#define YY_USE_CONST

#endif	/* defined (__STDC__) */
#endif	/* ! __cplusplus */

#ifdef YY_USE_CONST
#define yyconst const
#else
#define yyconst
#endif

#ifdef __cplusplus
extern "C" {
#endif
#ifndef YY_STRUCT_YY_BUFFER_STATE
#define YY_STRUCT_YY_BUFFER_STATE
struct yy_buffer_state
	{
	FILE *yy_input_file;

	char *yy_ch_buf;		/* input buffer */
	char *yy_buf_pos;		/* current position in input buffer */

	/* Size of input buffer in bytes, not including room for EOB
	 * characters.
	 */
	yy_size_t yy_buf_size;

	/* Number of characters read into yy_ch_buf, not including EOB
	 * characters.
	 */
	yy_size_t yy_n_chars;

	/* Whether we "own" the buffer - i.e., we know we created it,
	 * and can realloc() it to grow it, and should free() it to
	 * delete it.
	 */
	int yy_is_our_buffer;

	/* Whether this is an "interactive" input source; if so, and
	 * if we're using stdio for input, then we want to use getc()
	 * instead of fread(), to make sure we stop fetching input after
	 * each newline.
	 */
	int yy_is_interactive;

	/* Whether we're considered to be at the beginning of a line.
	 * If so, '^' rules will be active on the next match, otherwise
	 * not.
	 */
	int yy_at_bol;

    int yy_bs_lineno; /**< The line count. */
    int yy_bs_column; /**< The column count. */
    
	/* Whether to try to fill the input buffer when we reach the
	 * end of it.
	 */
	int yy_fill_buffer;

	int yy_buffer_status;

#define YY_BUFFER_NEW 0
#define YY_BUFFER_NORMAL 1
	/* When an EOF's been seen but there's still some text to process
	 * then we mark the buffer as YY_EOF_PENDING, to indicate that we
	 * shouldn't try reading from the input source any more.  We might
	 * still have a bunch of tokens to match, though, because of
	 * possible backing-up.
	 *
	 * When we actually see the EOF, we change the status to "new"
	 * (via yyrestart()), so that the user can continue scanning by
	 * just pointing yyin at a new input file.
	 */
#define YY_BUFFER_EOF_PENDING 2

	};
#endif /* !YY_STRUCT_YY_BUFFER_STATE */

#ifndef YY_TYPEDEF_YY_BUFFER_STATE
#define YY_TYPEDEF_YY_BUFFER_STATE
typedef struct yy_buffer_state *YY_BUFFER_STATE;
#endif
/* End block of stuff copied from param_lexer.c, which is created by params.l */

   YY_BUFFER_STATE yy_scan_bytes(yyconst char *bytes, yy_size_t len);
   int yyparse(void);
   int yylex(void);
   int yylex_destroy(void);

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

int pv_parseParameters(PV::PVParams * action_handler, const char * paramBuffer, size_t len)
{
   int result;
   handler = action_handler;
   yy_scan_bytes(paramBuffer, len);
   result = yyparse();
   yylex_destroy();
   return result;
}

%}

%union {char * sval; double dval; }
%token <sval> T_STRING
%token <sval> T_ID
%token <sval> T_ID_OVERWRITE
%token <dval> T_NUMBER
%token <sval> T_FILE_KEYWORD
%token <sval> T_FILENAME
%token <sval> T_INCLUDE
%token <sval> T_PARAM_SWEEP
%token <sval> T_BATCH_SWEEP

%%

declarations : /* empty */
             | declarations pvparams_directive
             | declarations parameter_group
             | declarations parameter_sweep
             | declarations batch_sweep
             ;


pvparams_directive : T_ID '=' T_NUMBER ';'
                         { handler->action_pvparams_directive($1, $3); }
                   /*
                   | T_ID_OVERWRITE '=' T_NUMBER ';'
                         { handler->action_pvparams_directive_overwrite($1, $3); }
                   */
                   ;

parameter_group_id : T_ID T_STRING '='
                      { handler->action_parameter_group_name($1, $2); }
                   ;

parameter_group : parameter_group_id '{' parameter_defs '}' ';'
                      { handler->action_parameter_group(); }
                   ;

                
parameter_defs : /* empty */
               | parameter_defs parameter_def
               | parameter_defs parameter_string_def
               | parameter_defs parameter_array_def
               | parameter_defs include_directive
               ;

parameter_array_def : T_ID '=' parameter_array ';'
                            { handler->action_parameter_array($1); }
                    | T_ID_OVERWRITE '=' parameter_array ';'
                            { handler->action_parameter_array_overwrite($1); }
                    ;

parameter_array : '[' parameter_array_values ']'

parameter_array_values : parameter_array_value
                       | parameter_array_values ',' parameter_array_value
                      ;

parameter_array_value : T_NUMBER
                         { handler->action_parameter_array_value($1); }
                      ;
                         

parameter_def : T_ID '=' T_NUMBER ';'
                 { handler->action_parameter_def($1, $3); }
              | T_ID_OVERWRITE '=' T_NUMBER ';'
                 { handler->action_parameter_def_overwrite($1, $3); }
              ;

parameter_string_def : T_ID '=' T_STRING ';'
                        { handler->action_parameter_string_def($1,$3); }
                     | T_ID '=' T_FILENAME ';'
                        { handler->action_parameter_filename_def($1,$3); }
                     | T_ID_OVERWRITE '=' T_STRING ';'
                        { handler->action_parameter_string_def_overwrite($1,$3); }
                     | T_ID_OVERWRITE '=' T_FILENAME ';'
                        { handler->action_parameter_filename_def_overwrite($1,$3); }
                     ;

/*include_directive : T_INCLUDE T_ID ';'*/
include_directive : T_INCLUDE T_STRING ';'
                        { handler->action_include_directive($2); }
                  ;


/* Sweeps */
parameter_sweep_id : T_PARAM_SWEEP T_STRING ':' T_ID '='
                      { handler->action_sweep_open($2, $4); }


parameter_sweep : parameter_sweep_id '{' parameter_sweep_values '}' ';'
                        { handler->action_parameter_sweep_close(); }
                ;

parameter_sweep_values : /* empty */
             | parameter_sweep_values_numbers
             | parameter_sweep_values_strings
             | parameter_sweep_values_filenames
             ;

parameter_sweep_values_numbers : parameter_sweep_values_number
                     | parameter_sweep_values_numbers parameter_sweep_values_number
                     ; /* empty not included because this leads to a reduce/reduce conflict if sweep_values is empty */

parameter_sweep_values_number : T_NUMBER ';'
                       { handler->action_parameter_sweep_values_number($1); }
                    ;

parameter_sweep_values_strings : parameter_sweep_values_string
                     | parameter_sweep_values_strings parameter_sweep_values_string
                     ; /* empty not included because this leads to a reduce/reduce conflict if sweep_values is empty */

parameter_sweep_values_string : T_STRING ';'
                       { handler->action_parameter_sweep_values_string($1); }
                    ;

parameter_sweep_values_filenames : parameter_sweep_values_filename
                       | parameter_sweep_values_filenames parameter_sweep_values_filename
                       ; /* empty not included because this leads to a reduce/reduce conflict if sweep_values is empty */

parameter_sweep_values_filename : T_FILENAME ';'
                         { handler->action_parameter_sweep_values_filename($1); }
                      ;

batch_sweep : batch_sweep_id '{' batch_sweep_values '}' ';'
                        { handler->action_batch_sweep_close(); }
                ;

batch_sweep_id : T_BATCH_SWEEP T_STRING ':' T_ID '='
                      { handler->action_sweep_open($2, $4); }
                   ;


batch_sweep_values : /* empty */
             | batch_sweep_values_numbers
             | batch_sweep_values_strings
             | batch_sweep_values_filenames
             ;

batch_sweep_values_numbers : batch_sweep_values_number
                     | batch_sweep_values_numbers batch_sweep_values_number
                     ; /* empty not included because this leads to a reduce/reduce conflict if sweep_values is empty */

batch_sweep_values_number : T_NUMBER ';'
                       { handler->action_batch_sweep_values_number($1); }
                    ;

batch_sweep_values_strings : batch_sweep_values_string
                     | batch_sweep_values_strings batch_sweep_values_string
                     ; /* empty not included because this leads to a reduce/reduce conflict if sweep_values is empty */

batch_sweep_values_string : T_STRING ';'
                       { handler->action_batch_sweep_values_string($1); }
                    ;

batch_sweep_values_filenames : batch_sweep_values_filename
                       | batch_sweep_values_filenames batch_sweep_values_filename
                       ; /* empty not included because this leads to a reduce/reduce conflict if sweep_values is empty */

batch_sweep_values_filename : T_FILENAME ';'
                         { handler->action_batch_sweep_values_filename($1); }
                      ;


                     
