from ply import lex, yacc
from collections import OrderedDict
from sklearn.cluster import KMeans
import numpy as np
import difflib
import docx
import re
import os

def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

def remove_comments(code):
    pattern = r'/\*.*?\*/'
    code = re.sub(pattern, '', code, flags=re.DOTALL)

    pattern = r'//.*?$'
    code = re.sub(pattern, '', code, flags=re.MULTILINE)

    # Remove lines starting with #include
    code = re.sub(r'^\s*#include.*?$', '', code, flags=re.MULTILINE)

    # Replace macro definitions and remove them
    code_lines = code.split('\n')
    for i, line in enumerate(code_lines):
        if line.startswith("#define"):
            parts = line.split()
            if len(parts) == 3 and parts[1].isidentifier():
                macro = parts[1]
                value = parts[2]
                # Replace occurrences of the macro in the code
                for j, line in enumerate(code_lines):
                    code_lines[j] = line.replace(macro, value)
            # Remove the macro definition from the code
            code_lines[i] = ""

    # Reconstruct the code without macro definitions
    code = "\n".join(code_lines)

    return code

Files = []
All_code = []
File_name = []
# "/Users/prem/Downloads/Excersise 6"
root = str(input("Enter folder path:"))
Similarity_matrix = []

for path, subdirs, files in os.walk(root):
    for name in files:
        Files.append(os.path.join(path, name))

for f in range(1,len(Files)):
    codes = []
    sample_text = getText(Files[f])

    # Define the regex pattern
    pattern = r"Code-(.*?)Sample Input and Output-"

    # Find all matches of the pattern using regex
    matches = re.findall(pattern, sample_text, re.DOTALL)

    # Print all matches
    for match in matches:
        extracted_text = match.strip()
        codes.append(remove_comments(extracted_text))
    All_code.append(codes.copy())
    parts = Files[f].split('/')
    File_name.append(parts[-1])

l = []

tokens = (
    'IDENTIFIER',
    'CONSTANT', 
    'STRING_LITERAL', 
    'SIZEOF',
    'PTR_OP',
    'INC_OP', 
    'DEC_OP',
    'LEFT_OP', 
    'RIGHT_OP', 
    'LE_OP', 
    'GE_OP', 
    'EQ_OP', 
    'NE_OP',
    'AND_OP', 
    'OR_OP', 
    'MUL_ASSIGN', 
    'DIV_ASSIGN', 
    'MOD_ASSIGN', 
    'ADD_ASSIGN',
    'SUB_ASSIGN', 
    'LEFT_ASSIGN', 
    'RIGHT_ASSIGN', 
    'AND_ASSIGN',
    'XOR_ASSIGN', 
    'OR_ASSIGN', 
    'TYPE_NAME',
    'TYPEDEF',
    'EXTERN', 
    'STATIC', 
    'AUTO', 
    'REGISTER',
    'CHAR', 
    'SHORT', 
    'INT', 
    'LONG', 
    'SIGNED', 
    'UNSIGNED', 
    'FLOAT', 
    'DOUBLE', 
    'CONST', 
    'VOLATILE', 
    'VOID',
    'STRUCT', 
    'UNION', 
    'ENUM', 
    'ELLIPSIS',
    'CASE', 
    'DEFAULT',
    'IF', 
    'ELSE', 
    'SWITCH', 
    'WHILE', 
    'DO', 
    'FOR', 
    'GOTO', 
    'CONTINUE', 
    'BREAK', 
    'RETURN',
    'SEMI_COLON',
    'LBRACE',
    'RBRACE',
    'COMMA',
    'ASSIGNMENT_OP',
    'LPARENTHESIS',
    'RPARENTHESIS',
    'LSQUARE_BRACKETS',
    'RSQUARE_BRACKETS',
    'DOT',
    'AMBERCENT',
    'NOT',
    'TILD',
    'MINUS',
    'PLUS',
    'ASTERIX',
    'DIVIDE',
    'MODULOUS',
    'L',
    'G',
    'POWER',
    'PIPE',
    'QUESTION',
    'COLON',
)

def t_AUTO(t):
    r"auto"
    return t

def t_BREAK(t): 
    r"break"
    return t
			
def t_CASE(t):
    r"case"
    return t
			
def t_CHAR(t):
    r"char"
    return t

def t_CONST(t):
    r"const"
    return t

def t_CONTINUE(t):
    r"continue"
    return t
		
def t_DEFAULT(t):
    r"default"
    return t
		
def t_DO(t):
    r"do"
    return t
	
def t_DOUBLE(t):
    r"double"
    return t
		
def t_ELSE(t):
    r"else"
    return t
			
def t_ENUM(t):
    r"enum"
    return t	
		
def t_EXTERN(t):
    r"extern"
    return t
		
def t_FLOAT(t):  
    r"float"
    return t
	
def t_FOR(t):
    r"for"
    return t
	
def t_GOTO(t):
    r"goto"
    return t
	
def t_IF(t):
    r"if"
    return t	

def t_INT(t):
    r"int"
    return t
	
def t_LONG(t):
    r"long"
    return t
	
def t_REGISTER(t):
    r"register"
    return t
		
def t_RETURN(t):
    r"return"
    return t
		
def t_SHORT(t):
    r"short"
    return t

def t_SIGNED(t):
    r"signed"
    return t

def t_SIZEOF(t):
    r"sizeof"
    return t
	
def t_STATIC(t): 
    r"static"
    return t
		
def t_STRUCT(t): 
    r"struct"
    return t
		
def t_SWITCH(t): 
    r"switch"
    return t
	
def t_TYPEDEF(t): 
    r"typedef"
    return t

def t_UNION(t):
    r"union"
    return t
	
def t_UNSIGNED(t):
    r"unsigned"
    return t
	
def t_VOID(t): 
    r"void"
    return t
	
def t_VOLATILE(t): 
    r"volatile"
    return t
	
def t_WHILE(t):
    r"while"
    return t

def t_IDENTIFIER(t):
    r'[a-zA-Z_][a-zA-Z_0-9]*'
    return t

def t_CONSTANT(t):
    r"(0[xX][a-fA-F0-9]+([uUlL]|[a-zA-Z_])*?|0[0-7]+([uUlL]|[a-zA-Z_])*?|[1-9][0-9]*([uUlL]|[a-zA-Z_])*?|[a-zA-Z_]?'(\\.|[^\\'])+'|[0-9]+([Ee][+-]?[0-9]+)?(f|F|l|[a-zA-Z_])?|[0-9]+\.[0-9]*([Ee][+-]?[0-9]+)?(f|F|l|[a-zA-Z_])?)"
    return t

def t_STRING_LITERAL(t):
    r'[a-zA-Z_]?\"(\\.|[^\\"])*\"'
    return t

def t_ELLIPSIS(t):
    r"\.\.\."
    return t	

def t_RIGHT_ASSIGN(t): 
    r">>="
    return t
			
def t_LEFT_ASSIGN(t): 
    r"<<="
    return t
			
def t_ADD_ASSIGN(t):
    r'\+='
    return t
	
def t_SUB_ASSIGN(t): 
    r'\-='
    return t
			
def t_MUL_ASSIGN(t):
    r'\*='
    return t
		
def t_DIV_ASSIGN(t): 
    r'\/='
    return t
		
def t_MOD_ASSIGN(t): 
    r'\%='
    return t
			
def t_AND_ASSIGN(t): 
    r'\&='
    return t
			
def t_XOR_ASSIGN(t):
    r'\^='
    return t
			
def t_OR_ASSIGN(t):
    r'\|='
    return t
		
def t_RIGHT_OP(t):
    r">>"
    return t
			
def t_LEFT_OP(t): 
    r"<<"
    return t
			
def t_INC_OP(t): 
    r'\+\+'
    return t
			
def t_DEC_OP(t): 
    r'\-\-'
    return t
		
def t_PTR_OP(t):
    r'\->'
    return t
		
def t_AND_OP(t): 
    r"&&"
    return t
			
def t_OR_OP(t): 
    r'\|\|'
    return t
			
def t_LE_OP(t):
    r"<="
    return t
			
def t_GE_OP(t): 
    r">="
    return t
			
def t_EQ_OP(t): 
    r"=="
    return t
			
def t_NE_OP(t): 
    r'\!='
    return t

def t_SEMI_COLON(t):
    r';'
    return t

def t_LBRACE(t):
    r'{|<%'
    return t

def t_RBRACE(t):
    r'}|%>'
    return t

def t_COMMA(t):
    r','
    return t

def t_ASSIGNMENT_OP(t):
    r'='
    return t

def t_LPARENTHESIS(t):
    r'\('
    return t

def t_RPARENTHESIS(t):
    r'\)'
    return t

def t_LSQUARE_BRACKETS(t):
    r'\[|<:'
    return t

def t_RSQUARE_BRACKETS(t):
    r'\]|:>'
    return t

def t_DOT(t):
    r'\.'
    return t

def t_AMBERCENT(t):  
    r'&'
    return t

def t_NOT(t):
    r'!'
    return t

def t_TILD(t):
    r'~'
    return t

def t_MINUS(t):
    r'-'
    return t

def t_PLUS(t):
    r'\+'
    return t

def t_ASTERIX(t):
    r'\*'
    return t

def t_DIVIDE(t):
    r'/'
    return t

def t_MODULOUS(t):
    r'%'
    return t

def t_L(t):
    r'<'
    return t

def t_G(t):
    r'>'
    return t 

def t_POWER(t):
    r'\^'
    return t

def t_PIPE(t):
    r'\|'
    return t

def t_QUESTION(t):
    r'\?'
    return t

def t_COLON(t):
    r':'
    return t

# Ignored characters
def t_IGNORE(t):
    r'[ \t\v\n\f]'
    return

# Error handling rule
def t_error(t):
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

def p_primary_expression(p):
    '''primary_expression   : IDENTIFIER
	                        | CONSTANT
	                        | STRING_LITERAL
	                        | LPARENTHESIS expression RPARENTHESIS 
    '''
    l.append(1)

def p_postfix_expression(p):
    '''
    postfix_expression  : primary_expression
	                    | postfix_expression LSQUARE_BRACKETS expression RSQUARE_BRACKETS
	                    | postfix_expression LPARENTHESIS RPARENTHESIS
	                    | postfix_expression LPARENTHESIS argument_expression_list RPARENTHESIS
	                    | postfix_expression DOT IDENTIFIER
	                    | postfix_expression PTR_OP IDENTIFIER
	                    | postfix_expression INC_OP
	                    | postfix_expression DEC_OP 
    '''
    l.append(2)

def p_argument_expression_list(p):
    '''
    argument_expression_list    : assignment_expression
	                            | argument_expression_list COMMA assignment_expression 
    '''
    l.append(3)

def p_unary_expression(p):
    '''
    unary_expression    : postfix_expression
	                    | INC_OP unary_expression
	                    | DEC_OP unary_expression
	                    | unary_operator cast_expression
	                    | SIZEOF unary_expression
	                    | SIZEOF LPARENTHESIS type_name RPARENTHESIS
    '''
    l.append(4)

def p_unary_operator(p):
    '''
    unary_operator  : AMBERCENT
	                | ASTERIX
	                | PLUS
	                | MINUS
	                | TILD
	                | NOT
    '''
    l.append(5)

def p_cast_expression(p):
    '''
    cast_expression : unary_expression
	                | LPARENTHESIS type_name RPARENTHESIS cast_expression 
    '''
    l.append(6)

def p_multiplicative_expression(p):
    '''
    multiplicative_expression   : cast_expression
	                            | multiplicative_expression ASTERIX cast_expression
	                            | multiplicative_expression DIVIDE cast_expression
	                            | multiplicative_expression MODULOUS cast_expression 
    '''
    l.append(7)

def p_additive_expression(p):
    '''
    additive_expression : multiplicative_expression
	                    | additive_expression PLUS multiplicative_expression
	                    | additive_expression MINUS multiplicative_expression 
    '''
    l.append(8)

def p_shift_expression(p):
    '''
    shift_expression    : additive_expression
	                    | shift_expression LEFT_OP additive_expression
	                    | shift_expression RIGHT_OP additive_expression 
    '''
    l.append(9)

def p_relational_expression(p):
    '''
    relational_expression   : shift_expression
	                        | relational_expression L shift_expression
	                        | relational_expression G shift_expression
	                        | relational_expression LE_OP shift_expression
	                        | relational_expression GE_OP shift_expression 
    '''
    l.append(10)

def p_equality_expression(p):
    '''
    equality_expression : relational_expression
	                    | equality_expression EQ_OP relational_expression
	                    | equality_expression NE_OP relational_expression 
    '''
    l.append(11)

def p_and_expression(p):
    '''
    and_expression  : equality_expression
	                | and_expression AMBERCENT equality_expression 
    '''
    l.append(12)

def p_exclusive_or_expression(p):
    '''
    exclusive_or_expression : and_expression
	                        | exclusive_or_expression POWER and_expression 
    '''
    l.append(13)

def p_inclusive_or_expression(p):
    '''
    inclusive_or_expression : exclusive_or_expression
	                        | inclusive_or_expression PIPE exclusive_or_expression 
    '''
    l.append(14)

def p_logical_and_expression(p):
    '''
    logical_and_expression  : inclusive_or_expression
	                        | logical_and_expression AND_OP inclusive_or_expression 
    '''
    l.append(15)

def p_logical_or_expression(p):
    '''
    logical_or_expression   : logical_and_expression
	                        | logical_or_expression OR_OP logical_and_expression 
    '''
    l.append(16)

def p_conditional_expression(p):
    '''
    conditional_expression  : logical_or_expression
	                        | logical_or_expression QUESTION expression COLON conditional_expression 
    '''
    l.append(17)

def p_assignment_operator(p):
    '''
    assignment_operator : ASSIGNMENT_OP
	                    | MUL_ASSIGN
	                    | DIV_ASSIGN
	                    | MOD_ASSIGN
	                    | ADD_ASSIGN
	                    | SUB_ASSIGN
	                    | LEFT_ASSIGN
	                    | RIGHT_ASSIGN
	                    | AND_ASSIGN
	                    | XOR_ASSIGN
	                    | OR_ASSIGN 
    '''
    l.append(18)

def p_assignment_expression(p):
    '''
    assignment_expression   : conditional_expression
	                        | unary_expression assignment_operator assignment_expression 
    '''
    l.append(19)

def p_expression(p):
    '''
    expression  : assignment_expression
	            | expression COMMA assignment_expression
    '''
    l.append(20)

def p_constant_expression(p):
    '''
    constant_expression : conditional_expression
    '''
    l.append(21)

def p_declaration(p):
    '''
    declaration : declaration_specifiers SEMI_COLON
	            | declaration_specifiers init_declarator_list SEMI_COLON
    '''
    l.append(22)

def p_declaration_specifiers(p):
    '''
    declaration_specifiers  : storage_class_specifier
	                        | storage_class_specifier declaration_specifiers
	                        | type_specifier
	                        | type_specifier declaration_specifiers
	                        | type_qualifier
	                        | type_qualifier declaration_specifiers
    '''
    l.append(23)

def p_init_declarator_list(p):
    '''
    init_declarator_list    : init_declarator
	                        | init_declarator_list COMMA init_declarator
    '''
    l.append(24)

def p_init_declarator(p):
    '''
    init_declarator : declarator
	                | declarator ASSIGNMENT_OP initializer
    '''
    l.append(25)

def p_storage_class_specifier(p):
    '''
    storage_class_specifier : TYPEDEF
	                        | EXTERN
	                        | STATIC
	                        | AUTO
	                        | REGISTER
    '''
    l.append(26)

def p_type_specifier(p):
    '''
    type_specifier  : VOID
	                | CHAR
	                | SHORT
	                | INT
	                | LONG
	                | FLOAT
	                | DOUBLE
	                | SIGNED
	                | UNSIGNED
	                | struct_or_union_specifier
	                | enum_specifier
	                | TYPE_NAME
    '''
    l.append(27)

def p_struct_or_union_specifier(p):
    '''
    struct_or_union_specifier   : struct_or_union IDENTIFIER LBRACE struct_declaration_list RBRACE
	                            | struct_or_union LBRACE struct_declaration_list RBRACE
	                            | struct_or_union IDENTIFIER
    '''
    l.append(28)

def p_struct_or_union(p):
    '''
    struct_or_union : STRUCT    
                    | UNION
    '''
    l.append(29)

def p_struct_declaration_list(p):
    '''
    struct_declaration_list : struct_declaration
	                        | struct_declaration_list struct_declaration
    '''
    l.append(30)

def p_struct_declaration(p):
    '''
    struct_declaration  : specifier_qualifier_list struct_declarator_list SEMI_COLON
    '''
    l.append(31)

def p_specifier_qualifier_list(p):
    '''
    specifier_qualifier_list    : type_specifier specifier_qualifier_list
	                            | type_specifier
	                            | type_qualifier specifier_qualifier_list
	                            | type_qualifier
    '''
    l.append(32)

def p_struct_declarator_list(p):
    '''
    struct_declarator_list  : struct_declarator
	                        | struct_declarator_list COMMA struct_declarator
    '''
    l.append(33)

def p_struct_declarator(p):
    '''
    struct_declarator   : declarator
	                    | COLON constant_expression
	                    | declarator COLON constant_expression
    '''
    l.append(34)

def p_enum_specifier(p):
    '''
    enum_specifier  : ENUM LBRACE enumerator_list RBRACE
	                | ENUM IDENTIFIER LBRACE enumerator_list RBRACE
	                | ENUM IDENTIFIER
    '''
    l.append(35)

def p_enumerator_list(p):
    '''
    enumerator_list : enumerator
	                | enumerator_list COMMA enumerator
    '''
    l.append(36)

def p_enumerator(p):
    '''
    enumerator  : IDENTIFIER
	            | IDENTIFIER ASSIGNMENT_OP constant_expression
    '''
    l.append(37)

def p_type_qualifier(p):
    '''
    type_qualifier  : CONST
	                | VOLATILE
    '''
    l.append(38)

def p_declarator(p):
    '''
    declarator  : pointer direct_declarator
	            | direct_declarator
    '''
    l.append(39)

def p_direct_declarator(p):
    '''
    direct_declarator   : IDENTIFIER
	                    | LPARENTHESIS declarator RPARENTHESIS
	                    | direct_declarator LSQUARE_BRACKETS constant_expression RSQUARE_BRACKETS
	                    | direct_declarator LSQUARE_BRACKETS RSQUARE_BRACKETS
	                    | direct_declarator LPARENTHESIS parameter_type_list RPARENTHESIS
	                    | direct_declarator LPARENTHESIS identifier_list RPARENTHESIS
	                    | direct_declarator LPARENTHESIS RPARENTHESIS
    '''
    l.append(40)

def p_pointer(p):
    '''
    pointer : ASTERIX
	        | ASTERIX type_qualifier_list
	        | ASTERIX pointer
	        | ASTERIX type_qualifier_list pointer
    '''
    l.append(41)

def p_type_qualifier_list(p):
    '''
    type_qualifier_list : type_qualifier
	                    | type_qualifier_list type_qualifier
    '''
    l.append(42)

def p_parameter_declaration(p):
    '''
    parameter_declaration   : declaration_specifiers declarator
	                        | declaration_specifiers abstract_declarator
	                        | declaration_specifiers
    '''
    l.append(43)

def p_parameter_list(p):
    '''
    parameter_list  : parameter_declaration
	                | parameter_list COMMA parameter_declaration
    '''
    l.append(44)

def p_parameter_type_list(p):
    '''
    parameter_type_list : parameter_list
	                    | parameter_list COMMA ELLIPSIS
    '''
    l.append(45)

def p_identifier_list(p):
    '''
    identifier_list : IDENTIFIER
	                | identifier_list COMMA IDENTIFIER
    '''
    l.append(46)

def p_type_name(p):
    '''
    type_name   : specifier_qualifier_list
	            | specifier_qualifier_list abstract_declarator
    '''
    l.append(47)

def p_abstract_declarator(p):
    '''
    abstract_declarator : pointer
	                    | direct_abstract_declarator
	                    | pointer direct_abstract_declarator
    '''
    l.append(48)

def p_direct_abstract_declarator(p):
    '''
    direct_abstract_declarator  : LPARENTHESIS abstract_declarator RPARENTHESIS
	                            | LSQUARE_BRACKETS RSQUARE_BRACKETS
	                            | LSQUARE_BRACKETS constant_expression RSQUARE_BRACKETS
	                            | direct_abstract_declarator LSQUARE_BRACKETS RSQUARE_BRACKETS
	                            | direct_abstract_declarator LSQUARE_BRACKETS constant_expression RSQUARE_BRACKETS
	                            | LPARENTHESIS RPARENTHESIS
	                            | LPARENTHESIS parameter_type_list RPARENTHESIS
	                            | direct_abstract_declarator LPARENTHESIS RPARENTHESIS
	                            | direct_abstract_declarator LPARENTHESIS parameter_type_list RPARENTHESIS
    '''
    l.append(49)

def p_initializer(p):
    '''
    initializer : assignment_expression
	            | LBRACE initializer_list RBRACE
	            | LBRACE initializer_list COMMA RBRACE
    '''
    l.append(50)

def p_initializer_list(p):
    '''
    initializer_list    : initializer
	                    | initializer_list COMMA initializer
    '''
    l.append(51)

def p_statement(p):
    '''
    statement   : labeled_statement
	            | compound_statement
	            | expression_statement
	            | selection_statement
	            | iteration_statement
	            | jump_statement
                | declaration
    '''
    l.append(52)

def p_labeled_statement(p):
    '''
    labeled_statement   : IDENTIFIER COLON statement
	                    | CASE constant_expression COLON statement
	                    | DEFAULT COLON statement
    '''
    l.append(53)

def p_compound_statement(p):
    '''
    compound_statement  : LBRACE RBRACE
	                    | LBRACE statement_list RBRACE
	                    | LBRACE declaration_list RBRACE
	                    | LBRACE declaration_list statement_list RBRACE
    '''
    l.append(54)

def p_declaration_list(p):
    '''
    declaration_list    : declaration
	                    | declaration_list declaration
    '''
    l.append(55)

def p_statement_list(p):
    '''
    statement_list  : statement
	                | statement_list statement
    '''
    l.append(56)

def p_expression_statement(p):
    '''
    expression_statement    : SEMI_COLON
	                        | expression SEMI_COLON
    '''
    l.append(57)

def p_selection_statement(p):
    '''
    selection_statement : IF LPARENTHESIS expression RPARENTHESIS statement
	                    | IF LPARENTHESIS expression RPARENTHESIS statement ELSE statement
	                    | SWITCH LPARENTHESIS expression RPARENTHESIS statement
    '''
    l.append(58)

def p_iteration_statement(p):
    '''
    iteration_statement : WHILE LPARENTHESIS expression RPARENTHESIS statement
	                    | DO statement WHILE LPARENTHESIS expression RPARENTHESIS SEMI_COLON
	                    | FOR LPARENTHESIS expression_statement expression_statement RPARENTHESIS statement
	                    | FOR LPARENTHESIS expression_statement expression_statement expression RPARENTHESIS statement
                        | FOR LPARENTHESIS declaration expression_statement expression RPARENTHESIS statement
                        | FOR LPARENTHESIS declaration expression_statement RPARENTHESIS statement
    '''
    l.append(59)

def p_jump_statement(p):
    '''
    jump_statement  : GOTO IDENTIFIER SEMI_COLON
	                | CONTINUE SEMI_COLON
	                | BREAK SEMI_COLON
	                | RETURN SEMI_COLON
	                | RETURN expression SEMI_COLON
    '''
    l.append(60)

def p_translation_unit(p):
    '''
    translation_unit    : external_declaration
	                    | translation_unit external_declaration
    '''
    l.append(61)

def p_external_declaration(p):
    '''
    external_declaration    : function_definition
	                        | declaration
    '''
    l.append(62)

def p_function_definition(p):
    '''
    function_definition : declaration_specifiers declarator declaration_list compound_statement
	                    | declaration_specifiers declarator compound_statement
	                    | declarator declaration_list compound_statement
	                    | declarator compound_statement
    '''
    l.append(63)

def p_error(p):
    l.append(-1)
    print("Syntax error at '%s'" % p.value)

def comparator(a,b):
    sm = difflib.SequenceMatcher(None,a,b)
    return sm.ratio()

for i in range(0,len(All_code[0])):
    l.clear()
    sim_score = []
    for j in range(0,len(All_code)):
        temp = []
        start = 'translation_unit'
        lexer.input(All_code[j][i])
        parser = yacc.yacc(start=start)
        parser.parse(All_code[j][i],lexer=lexer)
        a = l.copy()
        a.reverse()
        l.clear()
        for k in range(0,len(All_code)):
            start = 'translation_unit'
            lexer.input(All_code[k][i])
            parser = yacc.yacc(start=start)
            parser.parse(All_code[k][i],lexer=lexer)
            b = l.copy()
            b.reverse()
            if -1 in a or -1 in b:
                temp.append(0)
            else:
                temp.append(round(comparator(a,b),2))
            l.clear()
        sim_score.append(temp.copy())
    Similarity_matrix.append(sim_score.copy())

print()
print()

for i in range(0,len(Similarity_matrix)):
    for j in range(0,len(Similarity_matrix[i])):
        for k in range(0,len(Similarity_matrix[i][j])):
            Similarity_matrix[i][j][k] = round((Similarity_matrix[i][j][k] + Similarity_matrix[i][k][j])/2,2)
            Similarity_matrix[i][k][j] = Similarity_matrix[i][j][k]

grade = {}
Final_Grades = []
Final = []
Error_Students = []
Completely_copied = []
Unique = []
Most_Sim = []
Least_Sim = []
Med_Sim = []

for i in Similarity_matrix:
    k = 0
    Results = []
    err = []
    for j in i:
        if round(((sum(j)-1)/(len(j)-1)*100),2) < 0:
            err.append(File_name[k])
        else:
            Results.append((round(((sum(j)-1)/(len(j)-1)*100),2),File_name[k]))
        k += 1
    Results.sort()
    d = OrderedDict()
    for a in Results:
        if a[0] in d:
            d[a[0]].append(a[1])
        else:
            d[a[0]] = [a[1]]
    Final.append(d.copy())
    Error_Students.append(err.copy())

for i in Similarity_matrix:
    for j in i:
        print(j)
    print()

for i in File_name:
    print(i)
for i in Final:
    c = []
    u = []
    for j in i:
        if len(i[j]) > 1:
            c.append(i[j])
        else:
            u.append((j,i[j]))
    Completely_copied.append(c.copy())
    Unique.append(u.copy())

for i in Unique:
    values = [f[0] for f in i]
    cluster_centers = [[values[0]],[round((values[0]+values[-1])/2,2)],[values[-1]]]
    X = np.array(values).reshape(-1, 1)
    kmeans = KMeans(n_clusters=len(cluster_centers), init=np.array(cluster_centers))
    kmeans.fit(X)
    labels = kmeans.labels_
    k = 0
    ls = []
    meds = []
    ms = []
    for l in labels:
        if l == 0:
            ls.append(i[k][1])
        elif l == 1:
            meds.append(i[k][1])
        else:
            ms.append(i[k][1])
        k += 1
    Most_Sim.append(ms.copy())
    Least_Sim.append(ls.copy())
    Med_Sim.append(meds.copy())

print()

k = 0
for j in Error_Students:
    k += 1
    print("Student's Code which contain error for Question ",k,":")
    print(j)
    for i in j:
        if i not in grade:
            grade[i] = 0
    print()

Num_Ques = k

k = 0
for j in Completely_copied:
    k += 1
    print("Students who have completely copied Question ",k,":")
    print(j)
    for l in j:
        for i in l:
            if i in grade:
                grade[i] += 1
            else:
                grade[i] = 1
    print()

k = 0
for j in Least_Sim:
    k += 1
    print("Students who have Least similarity in their Code ",k,":")
    print()
    print(j) 
    for l in j:
        for i in l:
            if i in grade:
                grade[i] += 4
            else:
                grade[i] = 4 
    print()

k = 0
for j in Med_Sim:
    k += 1
    print("Students who have Moderate similarity in their Code ",k,":")
    print()
    print(j)
    for l in j:
        for i in l:  
            if i in grade:
                grade[i] += 3
            else:
                grade[i] = 3  
    print()

k = 0 
for j in Most_Sim:
    k += 1
    print("Students who have Most similarity in their Code ",k,":")
    print()
    print(j)
    for l in j:
        for i in l:    
            if i in grade:
                grade[i] += 2
            else:
                grade[i] = 2
    print()

for i in grade:
    grade[i] = round(grade[i]/Num_Ques,2)
    if grade[i] < 1:
        Final_Grades.append((i,"F"))
    elif grade[i] < 1.5:
        Final_Grades.append((i,"C"))
    elif grade[i] < 2:
        Final_Grades.append((i,"B"))
    elif grade[i] < 2.5:
        Final_Grades.append((i,"B+"))
    elif grade[i] < 3:
        Final_Grades.append((i,"A"))
    elif grade[i] < 3.5:
        Final_Grades.append((i,"A+"))
    else:
        Final_Grades.append((i,"O"))

width = max(len(e) for t in Final_Grades for e in t[:-1]) + 5
format=('%%-%ds' % width) * len(Final_Grades[0])
print('\n'.join(format % tuple(t) for t in Final_Grades))
print()
print()