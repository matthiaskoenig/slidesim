/* Headers for bindgen: the libCEED user API, plus free() for arrays libCEED allocates and
 * hands to the caller (CeedFree is not exported: libCEED allocates with calloc/malloc). */
#include <ceed.h>
#include <stdlib.h>
