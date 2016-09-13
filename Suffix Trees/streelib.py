#!/usr/bin/env python

'''
Keith Murray
'''


DEBUG = True
SILENT = False

# Data Structures 
class Node(object):
    sons		= 0 # a linked list of the sons of that node
    right_sibling	= 0 # a linked list of the right sibs
    left_sibling	= 0 # a linked list of the left sibs
    father		= 0 # a pointer to the father node
    suffix_link 	= 0 # a pointed to the node which reprsents the largest suffix of the current node
    path_position 	= 0 # an index of the start position of the nodes path
    edge_label_start 	= 0 # start index of the incomming edge
    edge_label_end 	= 0 # End index of the incoming edge  
    node_index 		= 0

    def __init__(self, father=0, path_position=0, edge_label_start=0, edge_label_end=0, node_index = -1):
	self.sons = 0 # a linked list of the sons of that node
	self.right_sibling = 0 # a linked list of the right sibs
	self.left_sibling = 0 # a linked list of the left sibs
	self.father = father # a pointer to the father node
	self.suffix_link = 0 # a pointed to the node which reprsents the largest suffix of the current node
	self.path_position = path_position # an index of the start position of the nodes path
	self.edge_label_start = edge_label_start # start index of the incomming edge
	self.edge_label_end = edge_label_end # End index of the incoming edge  
	self.node_index = node_index
    def __repr__(self):
	# Father
	if type(self.father) == int:
	    myFather = "()"
	else:
	    myFather = self.father.node_index
	    myFather = "(" + str(myFather) + ") "

	# Siblings
	if type(self.left_sibling) == int:
	    myLeft = "["
	else:
	    myLeft = self.left_sibling.node_index
	    myLeft = "[.." + str(myLeft) + ", "
	if type(self.right_sibling) == int:
	    myRight = "]"
	else:
	    myRight = self.right_sibling.node_index
	    myRight = ", " + str(myRight) + "..]"
	if type(self.suffix_link) == int:
	    myLink = "-X"
	else:
	    myLink = self.suffix_link.node_index
	    myLink = "-" + str(myLink) 
	mySelf = str(myLeft) + str(self.node_index) +str(myLink)+ str(myRight)
	# First Son
	if type(self.sons) == int:
	    mySon = "{}"
	else:
	    mySon = self.sons.node_index
	    mySon = "{" + str(mySon) + "..}"

	msg = "[" + str(myFather) + str(mySelf) + str(mySon) + "]"
	return msg


class SuffixTree(object):
    #ends = 0
    #tree_string = ""
    #length = 0
    #root = Node(0,0,0,0, 0)
    def __init__(self, tree_string=""):# = ""):
	self.ends = 0 # virtual end of all leaves
	self.tree_string = tree_string #The one and only real source string of the tree. All edge-labels
      					# contain only indices to this string and do not contain the characters
     					# themselves 
	self.length = len(tree_string) # length of the source string
	self.root = Node(0,0,0,0, 0) # the node that is the head of all others, no siblings or parent
	self.Nodes = [self.root] # This collects all nodes in tree 
    def __repr__(self):
	
	#msg = "A suffix tree for string length " + str(self.length) + " with " + str(len(self.Nodes)) + " nodes"
	return ST_PrintTree(self) # msg

class SuffixTreePath(object):
    # Basically an edge
    def __init__(self, begin=0, end=0):
	self.begin = begin
	self.end = end
    def __repr__(self):
	msg = "Path: " + str(begin) + ":" + str(end)
	return msg

class SuffixTreePos(object):
    def __init__(self, node=0, edge_pos=0):
	self.node = node
	self.edge_pos = edge_pos
    def __repr__(self):
	msg = "Current Position: \n\tNode:    \t" + str(self.node) + "\n\tEdge Pos:\t" + str(self.edge_pos)
	return msg

    

# ******************** #
# Internal functions

def find_son(tree, node, character):
    if (DEBUG == True):
	print "find_son" 
    '''
   find_son :
   Finds son of node that starts with a certain character. 

   Input : the tree, the node to start searching from and the character to be
           searched in the sons.
  
   Output: A pointer to the found son, 0 if no such son.
    '''
    #print tree.Nodes
    # gets first son
    #print node
    node = node.sons
    #print node, "**"
    #print tree.Nodes
    #print character
    #    scan all sons (all right siblings of the first son) for their first
    #    character (it has to match the character given as input to this function.
    #print tree.tree_string, "SLDKFJ", len(tree.tree_string)
    #print node.edge_label_start
    #print tree.tree_string[node.edge_label_start]
    while (type(node) != int)and node.edge_label_start<len(tree.tree_string) and tree.tree_string[node.edge_label_start] != character :
	#print character, node
	node = node.right_sibling
    return node

def get_node_label_end(tree, node):
    if (DEBUG == True):
	print "get_node_label_end" 
    '''
   get_node_label_end :
   Returns the end index of the incoming edge to that node. This function is
   needed because for leaves the end index is not relevant, instead we must look
   at the variable "e" (the global virtual end of all leaves). Never refer
   directly to a leaf's end-index.

   Input : the tree, the node its end index we need.

   Output: The end index of that node (meaning the end index of the node's
   incoming edge).
    '''
    # If it's a leaf - return e
    if (type(node.sons) == int):
	return tree.ends
    # If it's not a leaf - return its real end 
    return node.edge_label_end

def get_node_label_length(tree, node):
    if (DEBUG == True):
	print "get_node_label_length"
    '''
   get_node_label_length :
   Returns the length of the incoming edge to that node. Uses get_node_label_end
   (see above).

   Input : The tree and the node its length we need.

   Output: the length of that node.
    '''
    # Calculate and return the lentgh of the node
    return get_node_label_end(tree, node) - node.edge_label_start +1

def is_last_char_in_edge(tree, node, edge_pos):
    if (DEBUG == True):
	print "is_last_char_in_edge" 
    '''
   is_last_char_in_edge :
   Returns 1 if edge_pos is the last position in node's incoming edge.

   Input : The tree, the node to be checked and the position in its incoming
           edge.

   Output: the length of that node.
    '''
    if (edge_pos == get_node_label_length(tree, node)-1):
	return 1
    return 0

def connect_siblings(left_sib, right_sib):
    if (DEBUG == True):
	print "connect_siblings"
    '''
   connect_siblings :
   Connect right_sib as the right sibling of left_sib and vice versa.

   Input : The two nodes to be connected.

   Output: None.
    '''
    
    if (left_sib != 0):
	# Connect the right node as the right sibling of the left node
	left_sib.right_sibling = right_sib
    if (right_sib != 0):
	# Connect the left node as the left sibling of the right node
	right_sib.left_sibling = left_sib
    return left_sib, right_sib


def apply_extension_rule_2(tree, node, edge_label_begin, edge_label_end, path_pos, edge_pos, rule2_type):
    if (DEBUG == True):
	print "apply_extension_rule_2" 
    '''
   apply_extension_rule_2 :
   Apply "extension rule 2" in 2 cases:
   1. A new son (leaf 4) is added to a node that already has sons:
                (1)	       (1)
               /   \	 ->   / | \
              (2)  (3)      (2)(3)(4)

   2. An edge is split and a new leaf (2) and an internal node (3) are added:
              | 	  |
              | 	 (3)
              |     ->   / \
             (1)       (1) (2)

   Input : See parameters.

   Output: A pointer to the newly created leaf (new_son case) or internal node
   (split case).
    '''


    # THIS FUNCTION IS WHY THE TREE STRUCT WILL BE REDONE IN PYTHON
    #	Look at all the unsaved modifications!!! :P
    
    #new_leaf = Node()
    #new_internal = Node()
    #son = Node()
    if rule2_type == "new_son":
	# New Son
	# Create a new leaf (4) with the characters of the extension
	new_leaf = Node(node, edge_label_begin, edge_label_end, path_pos, len(tree.Nodes))
	# Connect new_leaf (4) as the new son of node (1)
	son = node.sons
	while (son.right_sibling != 0):
	    son = son.right_sibling
	son, newleaf = connect_siblings(son, new_leaf)
	# Return (4)
	return new_leaf
    # SPLIT
    # Create a new internal node (3) at the split point
    new_internal = Node(node.father, node.edge_label_start, node.edge_label_start+edge_pos, node.path_position)
    # Update the node (1) incoming edge starting index (it now starts where node
    #   (3) incoming edge ends)
    node.edge_label_start += edge_pos+1
    # Create a new leaf (2) with the characters of the extension
    new_leaf = Node(new_internal, edge_label_begin, edge_label_end, path_pos)
    
    # Connect new_internal (3) where node (1) was */
    # Connect (3) with (1)'s left sibling */
    node.left_sibling, new_internal = connect_siblings(node.left_sibling, new_internal)
    #  connect (3) with (1)'s right sibling
    new_internal, node.right_sibling = connect_siblings(new_internal, node.right_sibling)
    node.left_sibling = 0
    
    # Connect (3) with (1)'s father
    if (new_internal.father.sons == node):
	new_internal.father.sons = new_internal
    # Connect new_leaf (2) and node (1) as sons of new_internal (3)
    new_internal.sons = node
    node.father = new_internal
    node, new_leaf = connect_siblings(node, new_leaf)
    

    # Update tree.Nodes ?

    # return (3)
    return new_internal

def trace_single_edge(tree, node, keyStr, edge_pos, chars_found, skip_type, search_done):
    if (DEBUG == True):
	print "trace_single_edge"
    """
   trace_single_edge :
   Traces for a string in a given node's OUTcoming edge. It searches only in the
   given edge and not other ones. Search stops when either whole string was
   found in the given edge, a part of the string was found but the edge ended
   (and the next edge must be searched too - performed by function trace_string)
   or one non-matching character was found.

   Input : The string to be searched, given in indices of the main string.

   Output: (by value) the node where tracing has stopped.
           (by reference) the edge position where last match occured, the string
	   position where last match occured, number of characters found, a flag
	   for signaling whether search is done, and a flag to signal whether
	   search stopped at a last character of an edge.
    """

    

    # Set default return values
    search_done = 1
    edge_pos = 0
    # Search for the first character of the string in the outcoming edge of node
    print (keyStr.begin, len(tree.tree_string), keyStr.end) if (SILENT == False) else "",
    #print tree.Nodes
    cont_node = find_son(tree, node, tree.tree_string[keyStr.begin])

    if(cont_node == 0):
	print "No Next Node"
	# Search is done, string not found
	edge_pos = get_node_label_length(tree,node)-1
	chars_found = 0
	return node, edge_pos, chars_found, search_done
    # Found first character - prepare for continuing the search 
    node = cont_node
    length = get_node_label_length(tree,node)
    str_len = keyStr.end - keyStr.begin + 1
    
    #  Compare edge length and string length. 
    # If edge is shorter then the string being searched and skipping is
    # enabled - skip edge 
    if (skip_type == "skip"):
	if length <= str_len:
	    chars_found = length
	    edge_pos = length-1
	    if length < str_len	:
		#print length, chars_found, str_len, " <---"
		search_done = 0
	else:
	    chars_found = str_len
	    edge_pos = str_len - 1
	return node, edge_pos, chars_found, search_done
    else:
	# Find minimum out of edge length and string length, and scan it
	if str_len < length:
	    length = str_len
	edge_pos = 1
	chars_found = 1
	for i in range(1,length):
	    # Compare current characters of the string and the edge. If equal - 
	    #   continue
	    if (tree.tree_string[node.edge_label_start+edge_pos] != tree.tree_string[keyStr.begin+edge_pos]):
		edge_pos = edge_pos -1
		return node, edge_pos, chars_found, search_done
	    edge_pos += 1
	    chars_found += 1
	
    #  The loop has advanced *edge_pos one too much
    edge_pos = edge_pos -1
    if (chars_found < str_len):
	# Search is not done yet
	search_done = 0
    return node, edge_pos, chars_found, search_done

 
	
def trace_string(tree, node, keyStr, edge_pos, chars_found, skip_type):
    if (DEBUG == True):
	print "trace_string"
    '''
   trace_string :
   Traces for a string in the tree. This function is used in construction
   process only, and not for after-construction search of substrings. It is
   tailored to enable skipping (when we know a suffix is in the tree (when
   following a suffix link) we can avoid comparing all symbols of the edge by
   skipping its length immediately and thus save atomic operations - see
   Ukkonen's algorithm, skip trick).
   This function, in contradiction to the function trace_single_edge, 'sees' the
   whole picture, meaning it searches a string in the whole tree and not just in
   a specific edge.

   Input : The string, given in indice of the main string.

   Output: (by value) the node where tracing has stopped.
           (by reference) the edge position where last match occured, the string
	   position where last match occured, number of characters found, a flag
	   for signaling whether search is done.
    '''

    # This variable will be 1 when search is done.
    #  It is a return value from function trace_single_edge
    search_done = 0
    # This variable will hold the number of matching characters found in the
    #  current edge. It is a return value from function trace_single_edge
    chars_found = 0

    while search_done == 0:
	edge_pos = 0
	edge_chars_found = 0
	node, edge_pos, chars_found, search_done = trace_single_edge(tree, node, keyStr, edge_pos, edge_chars_found, skip_type, search_done)
	keyStr.begin += edge_chars_found
	chars_found += edge_chars_found
	print "SEARCH DONE", bool(search_done)
    return node, edge_pos, chars_found, search_done

def ST_findSubstring(tree, W, P):
    if (DEBUG == True):
	print "ST_findSubstring"
    '''
   ST_FindSubstring :
   Traces for a string in the tree. This function is used for substring search
   after tree construction is done. It simply traverses down the tree starting
   from the root until either the searched string is fully found ot one
   non-matching character is found. In this function skipping is not enabled
   because we don't know wether the string is in the tree or not (see function
   trace_string above).

   Input : The tree, the string W, and the length of W.

   Output: If the substring is found - returns the index of the starting
           position of the substring in the tree source string. If the substring
           is not found - returns ST_ERROR.
    '''
    
    # Starts with the root's son that has the first character of W as its
    #  incoming edge first characte
    node = find_son(tree, tree.root, W[0])
    k = 0
    j = 0
    ST_ERROR = "STREE ERROR"

    while (node !=0):
	k = node.edge_label_start
	node_label_end = get_node_label_end(tree, node)
	#  Scan a single edge - compare each character with the searched string
	while (j<P and k<=node_label_end and tree.tree_string[k] == W[j]):
	    j += 1
	    k += 1

	# Checking which of the stopping conditions are true
	if (j == P):
	    # W was found - it is a substring. Return its path starting index
	    return node.path_position;
	elif (k > node_label_end):
	    # Current edge is found to match, continue to next edge
	    node = find_son(tree, node, W[j])
	else:
	    # One non-matching symbols is found - W is not a substring 
	    return ST_ERROR
    return ST_ERROR

def follow_suffix_link(tree, pos):
    if (DEBUG == True):
	print "follow_suffix_link"
    '''
   follow_suffix_link :
   Follows the suffix link of the source node according to Ukkonen's rules. 

   Input : The tree, and pos. pos is a combination of the source node and the 
           position in its incoming edge where suffix ends.
   Output: The destination node that represents the longest suffix of node's 
           path. Example: if node represents the path "abcde" then it returns 
           the node that represents "bcde".
    '''
    
    # gama is the string between node and its father, in case node doesn't have
    #   a suffix link
    gama = SuffixTreePath()
    # dummy argument for trace_string function 
    chars_found = 0

    if (pos.node == tree.root):
	return pos

    # If node has no suffix link yet or in the middle of an edge - remember the
    #  edge between the node and its father (gama) and follow its father's suffix
    #  link (it must have one by Ukkonen's lemma). After following, trace down 
    #  gama - it must exist in the tree (and thus can use the skip trick - see 
    #  trace_string function description) 
    if(pos.node.suffix_link == 0) or is_last_char_in_edge(tree, pos.node, pos.edge_pos)==0:
	# If the node's father is the root, than no use following it's link (it 
        #  is linked to itself). Tracing from the root (like in the naive 
        #  algorithm) is required and is done by the calling function SEA uppon 
        #  recieving a return value of tree->root from this function 
	if (pos.node.father == tree.root):
	    pos.node = tree.root
	    return pos
	#  Store gama - the indices of node's incoming edge 
	gama.begin = pos.node.edge_label_start
	gama.end = pos.node.edge_label_start + pos.edge_pos
	#Follow father's suffix link 
	pos.node = pos.node.father.suffix_link
	# Down-walk gama back to suffix_link's son
	pos.node, edge_pos, chars_found, search_done = trace_string(tree, pos.node, gama, pos.edge_pos, chars_found, "skip")

    else:
	#  If a suffix link exists - just follow it 
	pos.node = pos.node.suffix_link
	pos.edge_pos = get_node_label_length(tree, pos.node)-1

    return pos


def create_suffix_link(node, link):
    if (DEBUG == True):
	print "create_suffix_link"
    '''
   create_suffix_link :
   Creates a suffix link between node and the node 'link' which represents its 
   largest suffix. The function could be avoided but is needed to monitor the 
   creation of suffix links when debuging or changing the tree.

   Input : The node to link from, the node to link to.

   Output: None.
    '''
    node.suffix_link = link
    return 



def SEA(tree, pos, keyStr, rule_applied, after_rule_3, suffixless):
    if (DEBUG == True):
	print "SEA"
    '''
   SEA :
   Single-Extension-Algorithm (see Ukkonen's algorithm). Ensure that a certain 
   extension is in the tree.

   1. Follows the current node's suffix link.
   2. Check whether the rest of the extension is in the tree.
   3. If it is - reports the calling function SPA of rule 3 (= current phase is 
      done).
   4. If it's not - inserts it by applying rule 2.

   Input : The tree, pos - the node and position in its incoming edge where 
           extension begins, str - the starting and ending indices of the 
           extension, a flag indicating whether the last phase ended by rule 3 
           (last extension of the last phase already existed in the tree - and 
           if so, the current phase starts at not following the suffix link of 
           the first extension).

   Output: The rule that was applied to that extension. Can be 3 (phase is done)
           or 2 (a new leaf was created).
    '''

    chars_found =0
    path_pos = SuffixTreePath(keyStr.begin)
    #tmp = Node()

    # Follow suffix link only if it's not the first extension after rule 3 was applied
    if (after_rule_3 == 0):
	pos = follow_suffix_link(tree, pos)

    # If node is root - trace whole string starting from the root, else - trace last character only 
    if (pos.node == tree.root):
	pos.node, edge_pos, chars_found, search_done = trace_string(tree, tree.root, keyStr, pos.edge_pos, chars_found, "no_skip")
    else:
	keyStr.begin = keyStr.end
	chars_found = 0

	 # Consider 2 cases:
         #  1. last character matched is the last of its edge 
	if (is_last_char_in_edge(tree, pos.node, pos.edge_pos)):
	    #  Trace only last symbol of str, search in the  NEXT edge (node)
	    tmp = find_son(tree, pos.node, tree.tree_string[keyStr.end])
	    if (tmp != 0):
		pos.node = tmp
		pos.edge_pos = 0
		chars_found = 1
	#  2. last character matched is NOT the last of its edge 
	else:
	    # Trace only last symbol of str, search in the CURRENT edge (node)
	    #print tree.tree_string[keyStr.end], keyStr.end, pos.node.edge_label_start + pos.edge_pos+1
	    #print tree.tree_string[pos.node.edge_label_start + pos.edge_pos+1]
	    if (pos.node.edge_label_start + pos.edge_pos+1 < len(tree.tree_string)) and (tree.tree_string[pos.node.edge_label_start + pos.edge_pos+1] == tree.tree_string[keyStr.end]):
		pos.edge_pos +=1
		chars_found = 1


    # If whole string was found - rule 3 applies 
    if (chars_found == keyStr.end - keyStr.begin +1):
	rule_applied = 3
	# If there is an internal node that has no suffix link yet (only one may 
        # exist) - create a suffix link from it to the father-node of the 
        # current position in the tree (pos)
	if (suffixless != 0):
	    create_suffix_link(suffixless, pos.node.father)
	    #  Marks that no internal node with no suffix link exists
	    suffixless = 0
	return tree, rule_applied, pos, suffixless, keyStr
    # If last char found is the last char of an edge - add a character at the 
    #  next edge
    if (is_last_char_in_edge(tree, pos.node, pos.edge_pos) or pos.node == tree.root):
	# Decide whether to apply rule 2 (new_son) or rule 1 
	if (pos.node.sons != 0):
	    # Apply extension rule 2 new son - a new leaf is created and returned 
            #  by apply_extension_rule_2
	    newNode = apply_extension_rule_2(tree, pos.node, keyStr.begin+chars_found, keyStr.end, path_pos, 0, "new_son")
	    rule_applied = 2
	    tree.Nodes.append(newNode)
	    # If there is an internal node that has no suffix link yet (only one 
            #  may exist) - create a suffix link from it to the father-node of the 
            #  current position in the tree (pos)
	    if (suffixless != 0):
		create_suffix_link(suffixless, pos.node)
		# Marks that no internal node with no suffix link exists 
		suffixless = 0

    else:
	# Apply extension rule 2 split - a new node is created and returned by 
        #  apply_extension_rule_2 
	tmp = apply_extension_rule_2(tree, pos.node, keyStr.begin+chars_found, keyStr.end, path_pos, pos.edge_pos, "split")
	tree.Nodes.append(tmp)
	if (suffixless != 0):
	    create_suffix_link(suffixless, tmp)
	    # Marks that no internal node with no suffix link exists 
	    suffixless = 0
	else:
	    # Mark tmp as waiting for a link 
	    suffixless = tmp
	#  Prepare pos for the next extension 
	pos.node = tmp
	rule_applied = 2

    return tree, rule_applied, pos, suffixless, keyStr


def SPA(tree, pos, phase, extension, repeated_extension, suffixless):
    if (DEBUG == True):
	print "SPA"
    '''
   SPA :
   Performs all insertion of a single phase by calling function SEA starting 
   from the first extension that does not already exist in the tree and ending 
   at the first extension that already exists in the tree. 

   Input :The tree, pos - the node and position in its incoming edge where 
          extension begins, the phase number, the first extension number of that
          phase, a flag signaling whether the extension is the first of this 
          phase, after the last phase ended with rule 3. If so - extension will 
          be executed again in this phase, and thus its suffix link would not be
          followed.

   Output:The extension number that was last executed on this phase. Next phase 
          will start from it and not from 1
    '''
    
    # No such rule (0). Used for entering the loop 
    rule_applied = 0
    keyStr = SuffixTreePath()

    # Leafs Trick: Apply implicit extensions 1 through prev_phase 
    tree.ends = phase+1

    # Apply explicit extensions untill last extension of this phase is reached 
    #  or extension rule 3 is applied once
    while(extension <= phase+1):
	keyStr.begin = extension
	keyStr.end = phase+1
	
	#  Call Single-Extension-Algorithm 
	tree, rule_applied, pos, suffixless, keyStr = SEA(tree, pos, keyStr, rule_applied, repeated_extension, suffixless)

	# Check if rule 3 was applied for the current extension
	if(rule_applied == 3):
	    # Signaling that the next phase's first extension will not follow a 
            # suffix link because same extension is repeated
	    repeated_extension = 1
	    break
	repeated_extension = 0
	extension += 1

    return tree, phase, pos, repeated_extension, extension, suffixless

def ST_CreateTree(keyStr):
    if (DEBUG == True):
	print "ST_CreateTree" 
    '''
   ST_CreateTree :
   Allocates memory for the tree and starts Ukkonen's construction algorithm by 
   calling SPA n times, where n is the length of the source string.

   Input : The source string and its length. The string is a sequence of 
           unsigned characters (maximum of 256 different symbols) and not 
           null-terminated. The only symbol that must not appear in the string 
           is $ (the dollar sign). It is used as a unique symbol by the 
           algorithm and is appended automatically at the end of the string (by 
           the program, not by the user!). The meaning of the $ sign is 
           connected to the implicit/explicit suffix tree transformation, 
           detailed in Ukkonen's algorithm.

   Output: A pointer to the newly created tree. Keep this pointer in order to 
           perform operations like search and delete on that tree. Obviously, no
	   de-allocating of the tree space could be done if this pointer is 
	   lost, as the tree is allocated dynamically on the heap.
    '''
    length = len(keyStr)
    tree = SuffixTree(keyStr)
    pos = SuffixTreePos(0,0)
    repeated_extension = 0


    tree.length = length +1
    keyStr = keyStr + "$"
    tree.tree_string = keyStr
    tree.root = Node(0,0,0,0,0)
    tree.root.suffix_link = 0

    # Initializing algorithm parameters
    extension = 2
    phase = 2

    # Allocating first node, son of the root (phase 0), the longest path node 
    tree.root.sons = Node(tree.root, 1, tree.length, 1, len(tree.Nodes))
    tree.Nodes.append(tree.root.sons)
    suffixless = 0
    pos.node = tree.root
    pos.edge_pos = 0
    
    print tree.length if (SILENT == False) else "",
    print "Hopes and dreams" if (SILENT == False) else "",
    #  Ukkonen's algorithm begins here 
    while phase < tree.length:
	#  Perform Single Phase Algorithm 
	print (phase, tree.tree_string[0:phase]) if (SILENT == False) else "",
	tree, phase, pos, repeated_extension, extension, suffixless = SPA(tree, pos, phase, extension, repeated_extension, suffixless)
	#print "Type ", type(tree), " Phase ", phase, " | ", tree.length
	phase += 1

    return tree


def test_Create_Tree():
    testStr = "alksdjflaksd"
    # WATCH ALL THE FAILURES :D
    myTree = ST_CreateTree(testStr)
    #print myTree
    return


    


def ST_PrintNode(tree, node1, depth):
    node2 = node1.sons
    d = depth
    start = node1.edge_label_start
    end = get_node_label_end(tree, node1)
    #print start, end

    #print "HERE", start

    if depth > 0:
	while d>1:
	    print ("|"),
	    d = d-1
	print("+"),
	# Print the node iteself
	#print start, end
	while (start <= end):
	    print(str(tree.tree_string[start-1])),
	    start += 1
	#print "\t\t\t(" + str(node1.edge_label_start) + "," + str(end) + " | "
	print("\n"),
    while(node2!=0):
	ST_PrintNode(tree,node2, depth+1)
	node2 = node2.right_sibling

def ST_PrintTree(tree):
    print "\n*******"
    print ("root\n"),
    ST_PrintNode(tree, tree.root, 0)
    print "*******"
    node = tree.root.sons
    while node != 0:
	start = node.edge_label_start
	end = get_node_label_end(tree, node)
	print node, tree.tree_string[start: end]
	node = node.right_sibling 
    return ""





# Run
test_Create_Tree()
	    

















# ******************** #
# Public functions

# create tree
# search tree

