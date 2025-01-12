# Root finding basic solvers

# import modules
import warnings
import numpy as np


# Bisection's Method
def bisection(func, x1, x2, tol=1.0e-15):
    """
    Uses the bisection method to find a value x between x1 and x2
    for which f(x) = 0, to within the tolerance given.
    Defalt tolerance: 1.0e-15.
    """
    
    JMAX = 100
    fmid = func(x2)
    f = func(x1)
    
    if f*fmid >= 0.: 
        warnings.warn('bisection: root must be bracketed')
        return 
        
    if f < 0.:
        root = x1
        dx = x2-x1
    else:
        root = x2
        dx = x1 - x2
        
    for j in range(1,JMAX+1):
        dx = dx*.5
        xmid = root + dx
        fmid = func(xmid)
        
        if fmid <= 0.:
            root = xmid
        
        # Convergence test or zero in midpoint: 
        if abs(dx) < abs(root)*tol or fmid == 0.0: 
            return root
        
    warnings.warn('bisection: too many bisections')
    
    return root
    
 
# Newton's Method  
def newton(f, df, x, tol=1.0e-15):
    """
    Uses Newton's method to find a value x near "x" for which f(x) = 0, 
    to within the tolerance given.
    
    INPUT: 
        Default tolerance: 1.0e-15
        Functions: f(x) and f'(x).
    """
    
    MAXIT = 20;
    root = x;
    
    for j in range(1, MAXIT + 1):
        
        dx = f(root) / df(root);
        root = root - dx;
        
        # Convergence is tested using relative error between
        # consecutive approximations.
        if (abs(dx) <= abs(root)*tol or f(root) == 0):
            return root
        
    warnings.warn('newton::err: max number of iterations reached.')  

    return root
    
    
# Brent's Method    
def brent(f, a, b, rel_tol=4.0e-16, abs_tol=2.e-12):
    """
    Brent's algorithm for root finding.
    
    Arguments:
    - f: The function for which to find the root.
    - a, b: Initial bracketing interval [a, b].
    - rel_tol: relative toleranece.
    - abs_tol: relative toleranece.
    - max_iterations: Maximum number of iterations.
    
    Returns:
    - The approximate root of the function.
    """

    fa = f(a)
    fb = f(b)
    c = a
    fc = fa
    e = b - a # interval size
    d = e     # step dx
    
    while (True):       
        # root between b and c
        # if root near c: move c,a to b, b to c 
        # else (root near b): don't do anything
        if (abs(fc) < abs(fb)):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa

        # compute tolerance
        tol = 2.0*rel_tol*abs(b) + abs_tol
        # half interval size
        m = 0.5*(c - b)

        # convergence
        if(abs(m) <= tol or fb == 0.0):
            break

        if (abs(e) < tol or abs(fa) <= abs(fb)):
            # take bisection
            e = m
            d = e # new step dx
        else:
            s = fb/fa
            if (a == c):
                # attemp linear interpolation
                p = 2.0*m*s
                q = 1.0 - s
            else:
                # attemp inverse quadratic interpolation
                q = fa/fc
                r = fb/fc
                p = s*(2.0*m*q*(q - r) - (b - a)*(r - 1.0))
                q = (q - 1.0)*(r - 1.0)*(s - 1.0)
                
            if(0.0 < p):
                q = -q
            else:
                p = -p

            s = e
                
            # check region of interpolation
            if (2.0*p < 3.0*m*q - abs(tol*q) and p < abs(0.5*s*q)):
                # accept interpolation
                e = d  # store previous d
                d = p/q # new step dx
            else:
                # use bisection
                e = m
                d = e # new step dx

        # move a to b (old root)
        a = b
        fa = fb

        # update b (new root)
        if (tol < abs(d)):
            # with step d
            b = b + d
        elif(0.0 < m):
            # min value of d is tol
            b = b + tol
        else:
            # min value of d is tol
            b = b - tol
        fb = f(b)

        # if root between b and a: move c to a
        # else (root between c and b): don't do anything
        if ((0.0 < fb and 0.0 < fc) or (fb <= 0.0 and fc <= 0.0)):
            c = a
            fc = fa
            e = b - c   # new interval size
            d = e       # new step dx

    return b    
    
    
    
    

    
#def funjac(x):
    # Placeholder: Define your system's functions and Jacobian matrix here
    #n = len(x)
    #fvec = np.zeros(n, dtype = np.float64)
    #fjac = np.zeros((n, n), dtype = np.float64)
    # Example definitions
    #fvec[0] = x[0] + x[1] - 6
    #fvec[1] = (x[0]**2) + (x[1])**2 - 20
    #fjac[0, 0] = 1
    #fjac[0, 1] = 1
    #fjac[1,0] = 2 * x[0]
    #fjac[1,1] = 2 * x[1]
    #return fvec, fjac

def is_singular(matrix):
    # A practical check using the condition number
    return np.linalg.cond(matrix) > 1e12  # or some other large threshold

def mnewt(funjac,ntrial, x, tolx, tolf):
    n = len(x)
    for k in range(ntrial):
        fvec, fjac = funjac(x)
        print(fvec)
        errf = np.sum(np.abs(fvec))
        if errf < tolf:
            return x, errx  # Converged in function value
        
            
        if is_singular(fjac):
            return x, "Jacobian is singular or nearly singular"
        p = np.linalg.solve(fjac, -fvec)  # Solve for the step directly
        errx = np.sum(np.abs(p))
        x += p
        if errx < tolx:
            return x, errx  # Converged in x increment

        

    return x  # Return the last attempted x

# Example usage:
#DONT FORGET TO DECLARE X INITIAL (AND ALL THE VARIABLES) AS NP.FLOAT64
#x_initial = np.array([2,2.80], dtype = np.float64)  # Initial guess
#tolx = 1e-14  # Tolerance in x
#tolf = 1e-14 # Tolerance in function value
#ntrial = 10  # Maximum number of trials
#x_solution = mnewt(funjac,ntrial, x_initial, tolx, tolf)
#print("Solution:", x_solution)



def newton_safe(f,df,x1,x2,rel_tol = 4.0e-16,abs_tol = 2.0e-12):
    
    N = 100 #max iteration
    
    #Validation: root must be bracketed
    f1 = f(x1)
    f2 = f(x2)
    
    if (f1*f2 > 0):
        print("Root must be bracketed")
        return None
    
    #We want too define fl st fl < 0 and fh st fh > 0. This definitions will help us to bracket the root more easy.
    
    if f1 < 0:
        fl = f1
        fh = f2
        xl = x1
        xh = x2
    
    else:
        fh = f1
        xh = x1
        fl = f2
        xl = x2
    
    # Initial guess
     
    if abs(fl) < abs(fh):
        root = xl
    else:
        root = xh
    
    
    #"Previous" step for the first step
    #dxold = (xh-xl)
    dx = (xh - xl) #Here dx means interval size
    fr = f(root)
    dfr = df(root)
    
    for i in range(1,N+1):
        #print()
        #print()
        #print("Iteration ",i)
        #print("dx: ", abs(dx))
        # Check if the new root proposed by newton methods lies in the safe interval
        nwt_out_range = ( ((root - xh) * dfr - fr) * ((root - xh) * dfr - fr) ) > 0
        
        #Check newton low convergence:  |f(new_root)| > 2 |f(root)|
        
        #nwt_slow = abs(2.0 * fr) > abs(dxold * dfr)
        nwt_slow = abs(2.0 * fr) > abs(dx * dfr)
        
        if (nwt_out_range and nwt_slow):
            #Use bisection: xl as point reference
            # We cant use root as point reference because root can be xl or xh
            
            #print("bisection")
            #dxold = dx
            dx = 0.5 * (xh-xl) #Here dx means step/displacement
            root = xl + dx
            
            #print("Newton would be: ", f(root - fr/dfr))
        
        else:
            #print("newton")
            #dxold = dx
            dx = -fr/dfr
            root = root + dx
            #print("step: ", dx)
        
        if (abs(dx) < abs(root) * rel_tol + abs_tol):
                #print("iterations: ",i)
                return root
            
        
        #Prepare the next iteration
        fr = f(root)
        
        #print("froot",(fr))
        
        dfr = df(root)
    
        #Mantain the defintions of xl and xh. This definitions help us to define more easy the new interval that bracket the root
        if (fr < 0):
            xl = root
            fl = fr
        else:
            xh = root
            fh = fr
    
        
        # Put the root such that fr = min{abs(fl),abs(fh)}
        if (abs(fl) < abs(fr)):
            root = xl
            fr = fl
        else:
            root = xh
            fr = fh
    
        
        dx = xl - xh #Here dx means interval size
        
    print("Max iterations reached")
    
    return root



def rtsafe(func, dfunc, x1, x2, xacc):
    MAXIT = 100
    fl = func(x1)
    fh = func(x2)
    if (fl > 0 and fh > 0) or (fl < 0 and fh < 0):
        raise ValueError("Root must be bracketed in rtsafe")
    
    if fl == 0:
        return x1
    if fh == 0:
        return x2
    
    if fl < 0:
        xl, xh = x1, x2
    else:
        xl, xh = x2, x1
    
    rts = 0.5 * (x1 + x2)
    dxold = abs(x2 - x1)
    dx = dxold
    f = func(rts)
    dfval = dfunc(rts)
    for j in range(MAXIT):
        if (((rts - xh) * dfval - f) * ((rts - xl) * dfval - f) > 0 or
            abs(2.0 * f) > abs(dxold * dfval)):
            dxold = dx
            dx = 0.5 * (xh - xl)
            rts = xl + dx
            if xl == rts:
                return rts
        else:
            dxold = dx
            dx = f / dfval
            temp = rts
            rts -= dx
            if temp == rts:
                return rts
        
        if abs(dx) < xacc:
            return rts
        
        f = func(rts)
        dfval = dfunc(rts)
        if f < 0:
            xl = rts
        else:
            xh = rts
    
    raise ValueError("Maximum number of iterations exceeded in rtsafe")
    
    
    

def rtsecsafe(func, x1, x2, xacc):
    MAXIT = 30
    fl = func(x1)
    f = func(x2)
    if abs(fl) < abs(f):
        x1, x2 = x2, x1
        fl, f = f, fl

    for j in range(MAXIT):
        dx = (x1 - x2) * f / (f - fl)
        x1, fl = x2, f
        x2 += dx
        f = func(x2)

        # If secant method goes out of bracket, use bisection
        if (x1 - x2) * dx > 0:
            x2 = x1 + (x1 - x2) / 2  # Bisection step

        if abs(dx) < xacc or f == 0.0:
            return x2

    raise ValueError("Maximum number of iterations exceeded in rtsec")
    

def rtsec(func, x1, x2, xacc):
    MAXIT = 100
    f1 = func(x1)
    f = func(x2)
    if abs(f1) < abs(f):
        x1, x2 = x2, x1
        f1, f = f, f1
    rts = x2

    for j in range(MAXIT):
        dx = (x1 - rts) * f / (f - f1)
        x1, f1 = rts, f
        rts += dx
        f = func(rts)
        if abs(dx) < xacc or f == 0.0:
            return rts

    raise ValueError("Maximum number of iterations exceeded in rtsec")
    
    
    
    
def newton_gq(f, df, x, tol=1.0e-15):
    """
    Uses Newton's method to find a value x near "x" for which f(x) = 0, 
    to within the tolerance given.
    
    INPUT: 
        Default tolerance: 1.0e-15
        Functions: f(x) and f'(x).
    """
    
    MAXIT = 20;
    root = x;
    
    for j in range(1, MAXIT + 1):
        
        dx = f(root) / df(root);
        root = root - dx;
        
        # Convergence is tested using relative error between
        # consecutive approximations.
        if (abs(dx) <= abs(root)*tol or f(root) == 0):
            return (root,df(root))
        
    warnings.warn('newton::err: max number of iterations reached.')  

    return (root,df(root))