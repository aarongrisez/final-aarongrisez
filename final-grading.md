# Final Grading Rubric
## Aaron Grisez

**Rubric:**
  - 0 : Preliminaries (29/30)
    - README : 5/5
    - Signature : 5/5
    - Travis : 5/5
    - Nosetests : 4/5
      - 2 tests, pass in 26sec
      - Rossler test only tests types, not correctness of algorithm. It would be better to work out at least one step of the diffeq by hand to ensure that the implementation matches what is expected. In this case the plots give good evidence of correctness, but one should also quantitatively check.
    - ```rossler.py``` : 5/5
      - Good organization. Good docstrings. Clean code.
    - Notebook : 5/5
      - Very nice!  A few minor suggestions: Put the import statements in a cell at the very top of the notebook (by convention). Name etc. need not be section headers. Use section headers in the rest of the notebook to block out actual sections. For presentations you can [suppress warnings](http://stackoverflow.com/questions/9031783/hide-all-warnings-in-ipython).
  - 1 : Rossler Implementation (26/26)
    - Clean implementation. Good use of numba where it can help. Cute use of for_loop (would jit help applied to integrate directly?).
  - 2 : Plotting Functions (14/14)
    - plotx : 2/2
    - ploty : 2/2
    - plotz : 2/2
    - plotxy : 2/2
    - plotyz : 2/2
    - plotxz : 2/2
    - plotxyz : 2/2
  - 3 : Chaos Exploration (14/15)
    - Good presentation of main results, and discussion. Note your list of equations at the beginning is missing two of the equations.
  - 4 : Local Maxima (15/15)
    - The displayed plot is correct.

**Total: (98/100)**