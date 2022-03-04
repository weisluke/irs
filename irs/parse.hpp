#pragma once

#include <string>


/*******************************************************
determine if a string exists within some array of char*s

\param begin -- starting pointer of array
\param end -- ending pointer of array
\param option -- string to find within the array

\returns bool -- true if string is in array, false if not
********************************************************/
bool cmd_option_exists(char** begin, char** end, const std::string& option);

/***************************************************************
determine if a string is a valid option from some options array

\param begin -- starting pointer of array of string options
\param end -- ending pointer of array of string options
\param option -- string to find within the array

\returns bool -- true if string was found in array, false if not
***************************************************************/
bool cmd_option_valid(const std::string* begin, const std::string* end, const std::string& option);

/**************************************************************
determine the parameter value of an option in array of char*s
assumed to be placed immediately after the option in some range

\param begin -- starting pointer of array
\param end -- ending pointer of array
\param option -- string to find the value of

\returns char* -- array of chars of the value after the option
**************************************************************/
char* cmd_option_value(char** begin, char** end, const std::string& option);

/**********************************************************************
determine if conversion of chars to double through std::strtod is valid

\param val -- the text to check

\returns bool -- true if text can be converted to double, false if not
**********************************************************************/
bool valid_double(char* val);

/*********************************************************************
determine if conversion of chars to float through std::strtof is valid

\param val -- the text to check

\returns bool -- true if text can be converted to float, false if not
*********************************************************************/
bool valid_float(char* val);

