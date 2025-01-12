# Polynomial interpolation

# import modules
import numpy as np 
import warnings 


# Lagrange Interpolation
def lagrange(x, y, xx):
    '''Computes the interpolated value at xx for the discrete 
    function given by (x, y) pairs using Lagrange interpolation.
    INPUT:
        x: abcisas of function to interpolate
        y: ordinates of function to interpolate 
        xx array or scalar to interpolate'''
    
    # Convert to numpy arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xx = np.asarray(xx, dtype=np.float64)
    
    # Convert scalar to array
    scalar = False
    if xx.ndim == 0:
        xx = xx[None]  # add new axis
        scalar = True
    
    n = len(x)
    nn = len(xx)
    sum = np.zeros_like(xx)
    
    for i in range(0, n):
        l = np.ones(nn)
        for j in range(0, n):
            if (i != j):
                l = l * (xx - x[j]) / (x[i] - x[j])
        sum = sum + y[i] * l
    
    # Scalar case
    if scalar:
        return sum[0]
    
    return sum



# Piecewise intepolation

# Spline function: computes second derivatives
def spline(x, y, yp1=None, ypn=None):
    """
    spline computes the second derivatives at all points x of the discrete
    function y(x) to be used for spline interpolation.
    INPUT:
        x: abscisas of discrete function
        y: ordinates of discrete function
        yp1: first derivative at first point, None for Natural SPLINE
        ypn: first derivative at last point, None for Natural SPLINE
    OUTPUT
        y2: second derivatives at all points
    """      
    
    # Convert to numpy arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
        
    # Initialize internal numpy arrays
    u = np.empty_like(x)
    y2 = np.empty_like(x)
        
    # Condition for the initial point    
    if (yp1 == None):
        # natural spline
        y2[0] = 0.
        u[0] = 0.
    else:
        # first derivative
        y2[0] = -0.5
        u[0]  = (3./(x[1] - x[0]))*((y[1] - y[0])/(x[1] - x[0]) - yp1)
    
    # Condition for the last point 
    if (ypn == None):
        # natural spline
        qn = 0.
        un = 0.
    else:
        # first derivative
        qn = 0.5
        un = (3./(x[n-1] - x[n-2]))*(ypn - (y[n-1] - y[n-2])/(x[n-1] - x[n-2]))
    
    # Setup tridiagonal equations
    for i in range(1, n-1):
        sig = (x[i] - x[i-1])/(x[i+1] - x[i-1])
        
        p = sig*y2[i-1] + 2.
        
        y2[i] = (sig - 1.)/p
        
        u[i] = (6.*((y[i+1] - y[i])/(x[i+1] - x[i]) - (y[i]-y[i-1])/
         (x[i] - x[i-1]))/(x[i+1] - x[i-1]) - sig*u[i-1])/p
    
    
    # Solve tridiagonal system for second derivatives
    y2[n-1] = (un - qn*u[n-2])/(qn*y2[n-2] + 1.)
    
    for k in range(n - 2, -1, -1):
        y2[k] = y2[k]*y2[k+1] + u[k]
        
    return y2
   
   
# Splint function: interpolate at a given point
def splint(xa, ya, y2a, x):
    """
    splint makes the spline interpolation of discrete function y(x) using 
    the second derivatives computed with spline algorithm.
    INPUT:
        x: abscisas of discrete function
        y: ordinates of discrete function
        y2: second derivatives at all points 
    OUTPUT:
        x: interpolated value
    """    
    
    # Convert to numpy arrays
    xa = np.asarray(xa, dtype=np.float64)
    ya = np.asarray(ya, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
        
    # Convert scalar to array
    scalar = False
    if x.ndim == 0:
        x = x[None]  # add new axis
        scalar = True
    
    # Allocate y
    y = np.empty_like(x)
    
    n = len(xa)
    for i in range(len(x)):
        # Find interval that contains x
        klo = 0
        khi = n - 1
        while (khi - klo > 1):
            k = int((khi + klo)/2)
            if(xa[k] > x[i]):
                khi = k
            else:
                klo = k
        
        # Evaluate cubic polynomial in x
        h = xa[khi] - xa[klo]
        
        if (h == 0.): 
            warnings.warn('splint::err: all xa must be different')
            return 0
        
        a = (xa[khi] - x[i])/h  # lagrange A(x)
        b = (x[i] - xa[klo])/h  # Lagrange B(x)
        
        y[i] = a*ya[klo] + b*ya[khi] + ((a**3 - a)*y2a[klo] + (b**3 - b)*y2a[khi])*(h**2)/6.
    
    # Scalar case
    if scalar:
        return y[0]
    
    return y
      
      
# NEVILLE INTERPOLATION


def neville_optimized(x_points, y_points, target_x):
    num_points = len(x_points)
    D = np.array(y_points.copy(), dtype = np.float64)  # U represents near differences
    U = np.array(y_points.copy(), dtype = np.float64)  # D represents far differences
    nearest_index = 0
    last_diff = 0.0  # Last difference used for error estimation
    
    # Find the index of the closest point in x_points to target_x
    min_diff = np.abs(target_x - x_points[0])
    for i in range(num_points):
        current_diff = np.abs(target_x - x_points[i])
        if current_diff < min_diff:
            nearest_index = i
            min_diff = current_diff
    
    
    
    
    
    
    
    
    interpolated_y = y_points[nearest_index]
    use_U = target_x > x_points[nearest_index]
    
    if nearest_index == 0:
        use_U = False
    if nearest_index == num_points - 1:
        use_U = True
        
    
    index = nearest_index

    for order in range(1, num_points):
        
        #Calculation of the D's and U's of the column m
        for i in range(num_points - order):
            x_diff_near = x_points[i] - target_x
            x_diff_far = x_points[i + order] - target_x
            w = D[i + 1] - U[i] #we use the U's and D's . Example: new U_i = old D_{i+1} - old U_i
            denominator = x_diff_near - x_diff_far
            if denominator == 0.0:
                raise Exception("Zero denominator encountered in neville_interpolation")
            # Update U and D
            denominator = w / denominator
            U[i] = x_diff_far * denominator
            D[i] = x_diff_near * denominator

        
        if use_U == True:
            
            index -= 1
            #print("U",index,order)
            last_diff = U[index]
            
            #Alternate to D
            use_U = False
            
            #Check if i can go down
            if index > (num_points - order - 1) -1:
                use_U = True
            
        else:
            #print("D",index,order)
            last_diff = D[index]
            
            #Alternate to U
            use_U = True
            
            #Check if I can go up
            if index - 1 < 0:
                use_U = False
        
        # Choose the appropriate difference based on the "straightest line" path
        #if 2 * (nearest_index + 1) < (num_points - order):
            #last_diff = D[nearest_index + 1]
        #else:
            #last_diff = U[nearest_index]
            #nearest_index -= 1
        #print("High index", len(U)-1-order)
        interpolated_y += last_diff
    
    
    return interpolated_y, abs(last_diff)



def neville_optimized_vec(x_points, y_points, X):
    
    x_points = np.asarray(x_points, dtype = np.float64)
    y_points = np.asarray(y_points, dtype = np.float64)
    X = np.asarray(X, dtype = np.float64)
    
    scalar = False
    
    if X.ndim == 0:
        X = X[None]
        scalar = True
    
    num_points = len(x_points)
    num_targets = len(X)
    
    F = np.zeros((num_targets,2), dtype = np.float64)
    
    for j in range(num_targets):
        target_x = X[j]
        D = np.array(y_points.copy(), dtype = np.float64)  # U represents near differences
        U = np.array(y_points.copy(), dtype = np.float64)  # D represents far differences
        nearest_index = 0
        last_diff = 0.0  # Last difference used for error estimation

        # Find the index of the closest point in x_points to target_x
        min_diff = np.abs(target_x - x_points[0])
        for i in range(num_points):
            current_diff = np.abs(target_x - x_points[i])
            if current_diff < min_diff:
                nearest_index = i
                min_diff = current_diff








        interpolated_y = y_points[nearest_index]
        use_U = target_x > x_points[nearest_index]

        if nearest_index == 0:
            use_U = False
        if nearest_index == num_points - 1:
            use_U = True


        index = nearest_index

        for order in range(1, num_points):

            #Calculation of the D's and U's of the column m
            for i in range(num_points - order):
                x_diff_near = x_points[i] - target_x
                x_diff_far = x_points[i + order] - target_x
                w = D[i + 1] - U[i] #we use the U's and D's 
                denominator = x_diff_near - x_diff_far
                if denominator == 0.0:
                    raise Exception("Zero denominator encountered in neville_interpolation")
                # Update U and D
                denominator = w / denominator
                U[i] = x_diff_far * denominator
                D[i] = x_diff_near * denominator


            if use_U == True:

                index -= 1
                #print("U",index,order)
                last_diff = U[index]

                #Alternate to D
                use_U = False

                #Check if i can go down
                if index > (num_points - order - 1) -1:
                    use_U = True

            else:
                #print("D",index,order)
                last_diff = D[index]

                #Alternate to U
                use_U = True

                #Check if I can go up
                if index - 1 < 0:
                    use_U = False

            # Choose the appropriate difference based on the "straightest line" path
            #if 2 * (nearest_index + 1) < (num_points - order):
                #last_diff = D[nearest_index + 1]
            #else:
                #last_diff = U[nearest_index]
                #nearest_index -= 1
            #print("High index", len(U)-1-order)
            interpolated_y += last_diff
            
        F[j,:] = (interpolated_y,abs(last_diff))
    
    if scalar:
        return F[0,:]
    
    return F



#HERMITE INTERPOLATION

def hermite_inter(x_points, y_points, dy_points, x_to_interpolate):
    """
    Computes the interpolated value at x_to_interpolate for the discrete 
    function given by (x_points, y_points) pairs and their derivatives 
    using Hermite interpolation.
    
    Args:
        x_points (list or numpy.array): The x-coordinates (abscissas) of the data points.
        y_points (list or numpy.array): The y-coordinates (ordinates) of the data points.
        dy_points (list or numpy.array): The derivatives at the data points.
        x_to_interpolate (scalar or numpy.array): The x-coordinate(s) at which to interpolate.
    
    Returns:
        numpy.array: The interpolated value(s) at x_to_interpolate.
    """
    
    x_points = np.asarray(x_points, dtype=np.float64)
    y_points = np.asarray(y_points, dtype=np.float64)
    dy_points = np.asarray(dy_points, dtype=np.float64)
    x_to_interpolate = np.asarray(x_to_interpolate, dtype=np.float64)
    
    num_data_points = len(x_points)
    interpolation_values = np.zeros_like(x_to_interpolate)
    
    for i in range(num_data_points):
        L = np.ones_like(x_to_interpolate)
        L_deriv = np.zeros_like(x_to_interpolate)
        for j in range(num_data_points):
            if i != j:
                factors = (x_to_interpolate - x_points[j]) / (x_points[i] - x_points[j])
                L *= factors
                L_deriv += 1.0 / (x_points[i] - x_points[j])
        
        #L_deriv is calculated at x_j
        #L is calculated at x_to_interpolate
        
        H = (1 - 2 * (x_to_interpolate - x_points[i]) * L_deriv) * L ** 2
        G = (x_to_interpolate - x_points[i]) * L ** 2
        
        interpolation_values += y_points[i] * H + dy_points[i] * G
    
    return interpolation_values






#SECOND NEVILLE INTERPOLATION


def neville_optimized_2(x_points, y_points, target_x):
    num_points = len(x_points)
    D = np.array(y_points.copy())  # U represents near differences
    U = np.array(y_points.copy())  # D represents far differences
    nearest_index = 0
    last_diff = 0.0  # Last difference used for error estimation
    
    
    
    # Find the index of the closest point in x_points to target_x
    min_diff = np.abs(target_x - x_points[0])
    for i in range(num_points):
        current_diff = np.abs(target_x - x_points[i])
        if current_diff < min_diff:
            nearest_index = i
            min_diff = current_diff
    
    
    
    
    
    
    
    
    interpolated_y = y_points[nearest_index]
    nearest_index -= 1

    for order in range(1, num_points):
        
        #Calculation of the D's and U's of the column m
        for i in range(num_points - order):
            x_diff_near = x_points[i] - target_x
            x_diff_far = x_points[i + order] - target_x
            w = D[i + 1] - U[i] #we use the U's and D's 
            denominator = x_diff_near - x_diff_far
            if denominator == 0.0:
                raise Exception("Zero denominator encountered in neville_interpolation")
            # Update U and D
            denominator = w / denominator
            U[i] = x_diff_far * denominator
            D[i] = x_diff_near * denominator

        
        
        
        # Choose the appropriate difference based on the "straightest line" path
        if 2 * (nearest_index + 1) < (num_points - order):
            last_diff = D[nearest_index + 1]
            #print("D", nearest_index+1)
        else:
            #print("U", nearest_index)
            last_diff = U[nearest_index]
            nearest_index -= 1
            
        
        interpolated_y += last_diff
    
    return interpolated_y, abs(last_diff)


def neville_optimized_vec2(x_points, y_points, X):
    
    x_points = np.asarray(x_points, dtype = np.float64)
    y_points = np.asarray(y_points, dtype = np.float64)
    X = np.asarray(X, dtype = np.float64)
    
    scalar = False
    
    if X.ndim == 0:
        X = X[None]
        scalar = True
    
    num_points = len(x_points)
    num_targets = len(X)
    
    F = np.zeros((num_targets,2), dtype = np.float64)
    
    
    
    for j in range(num_targets):
        
        target_x = X[j]
        D = np.array(y_points.copy())  # U represents near differences
        U = np.array(y_points.copy())  # D represents far differences
        nearest_index = 0
        last_diff = 0.0  # Last difference used for error estimation

        



        # Find the index of the closest point in x_points to target_x
        min_diff = np.abs(target_x - x_points[0])
        for i in range(num_points):
            current_diff = np.abs(target_x - x_points[i])
            if current_diff < min_diff:
                nearest_index = i
                min_diff = current_diff








        interpolated_y = y_points[nearest_index]
        nearest_index -= 1

        for order in range(1, num_points):

            #Calculation of the D's and U's of the column m
            for i in range(num_points - order):
                x_diff_near = x_points[i] - target_x
                x_diff_far = x_points[i + order] - target_x
                w = D[i + 1] - U[i] #we use the U's and D's 
                denominator = x_diff_near - x_diff_far
                if denominator == 0.0:
                    raise Exception("Zero denominator encountered in neville_interpolation")
                # Update U and D
                denominator = w / denominator
                U[i] = x_diff_far * denominator
                D[i] = x_diff_near * denominator




            # Choose the appropriate difference based on the "straightest line" path
            if 2 * (nearest_index + 1) < (num_points - order):
                last_diff = D[nearest_index + 1]
                #print("D", nearest_index+1)
            else:
                #print("U", nearest_index)
                last_diff = U[nearest_index]
                nearest_index -= 1


            interpolated_y += last_diff
        F[j,:] = (interpolated_y,abs(last_diff))
    
    
    if scalar:
        return F[0]
    
    return F




# HERMITE CUBIC SPLINE

import numpy as np

def calculate_hermite_derivatives(x, y):
    """
    Calculates the derivatives at each data point for the Hermite cubic spline interpolation.
    
    Args:
        x (numpy.array): The x-coordinates of the data points.
        y (numpy.array): The y-coordinates of the data points.
    
    Returns:
        numpy.array: The derivatives at each data point.
    """
    n = len(x)
    h = np.diff(x)             # Step size between points
    delta = np.diff(y) / h     # Slope between points
    d = np.zeros(n)            # Derivatives at the data points

    # Calculate the derivatives at the internal points
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] <= 0:
            d[k] = 0
        else:
            weight_1 = 2 * h[k] + h[k - 1]
            weight_2 = h[k] + 2 * h[k - 1]
            d[k] = (weight_1 + weight_2) / (weight_1 / delta[k - 1] + weight_2 / delta[k])
    
    # Calculate derivatives at the boundaries
    # Using the three-point formula for f'(x_0) and f'(x_n)
    d[0] = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    d[n - 1] = ((2 * h[n - 2] + h[n - 3]) * delta[n - 2] - h[n - 2] * delta[n - 3]) / (h[n - 2] + h[n - 3])
    
    
    
    if np.sign(d[0]) != np.sign(delta[0]):
            d[0] = 0
    elif np.sign(delta[0]) == np.sign(delta[1]) and abs(d[0]) > abs(3 * delta[0]):
            d[0] = 3 * delta[0]
            
    if np.sign(d[-1]) != np.sign(delta[-1]):
            d[-1] = 0
    elif np.sign(delta[-1]) == np.sign(delta[-2]) and abs(d[-1]) > abs(3 * delta[-1]):
            d[-1] = 3 * delta[-1]
            
    #print(d)
    return d


def hermite_derivatives_modified(x, y, der_type = None):
    """
    Calculates the derivatives at each data point for the Hermite cubic spline interpolation.
    
    Args:
        x (numpy.array): The x-coordinates of the data points.
        y (numpy.array): The y-coordinates of the data points.
    
    Returns:
        numpy.array: The derivatives at each data point.
    """
    n = len(x)
    h = np.diff(x)             # Step size between points
    delta = np.diff(y) / h     # Slope between points
    d = np.zeros(n)            # Derivatives at the data points

    # Calculate the derivatives at the internal points
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] <= 0:
            d[k] = 0
        else:
            weight_1 = 2 * h[k] + h[k - 1]
            weight_2 = h[k] + 2 * h[k - 1]
            
            if der_type == None or der_type =='harmonic':
                d[k] = (weight_1 + weight_2) / (weight_1 / delta[k - 1] + weight_2 / delta[k])
            elif der_type =='arithmetic':
                d[k] = (weight_1 * delta[k - 1] + weight_2 * delta[k])/(weight_1 + weight_2)
                
                
                #(weight_1 + weight_2) / (weight_1 / delta[k - 1] + weight_2 / delta[k])
    
    # Calculate derivatives at the boundaries
    # Using the three-point formula for f'(x_0) and f'(x_n)
    d[0] = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    d[n - 1] = ((2 * h[n - 2] + h[n - 3]) * delta[n - 2] - h[n - 2] * delta[n - 3]) / (h[n - 2] + h[n - 3])
    
    
    
    if np.sign(d[0]) != np.sign(delta[0]):
            d[0] = 0
    elif np.sign(delta[0]) == np.sign(delta[1]) and abs(d[0]) > abs(3 * delta[0]):
            d[0] = 3 * delta[0]
            
    if np.sign(d[-1]) != np.sign(delta[-1]):
            d[-1] = 0
    elif np.sign(delta[-1]) == np.sign(delta[-2]) and abs(d[-1]) > abs(3 * delta[-1]):
            d[-1] = 3 * delta[-1]
    
    #print(d)
    return d



def hcs(xa, ya, x, der_type = None):
    """
    Performs Hermite cubic spline interpolation for a given set of points.
    INPUT:
        xa: abscissas of the discrete function
        ya: ordinates of the discrete function
    OUTPUT:
        y: interpolated values at x
    """    
    
    # Convert to numpy arrays
    xa = np.asarray(xa, dtype=np.float64)
    ya = np.asarray(ya, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
        
    # Calculate derivatives using the Hermite derivative calculation
    derivatives = hermite_derivatives_modified(xa, ya, der_type)
    
    # Convert scalar to array
    scalar = False
    if x.ndim == 0:
        x = np.array([x])  # add new axis
        scalar = True
    
    # Allocate y for the results
    y = np.empty_like(x)
    
    n = len(xa)
    for i in range(len(x)):
        # Find the right place in the table by means of bisection
        # This is optimal if sequential calls to this function are at random values of x
        klo = 0
        khi = n - 1
        while khi - klo > 1:
            k = (khi + klo) >> 1
            if xa[k] > x[i]:
                khi = k
            else:
                klo = k
        
        # Compute cubic Hermite polynomial coefficients directly
        h = xa[khi] - xa[klo]
        t = (x[i] - xa[klo]) / h
        t2 = t * t
        t3 = t2 * t
        h00 = 2*t3 - 3*t2 + 1
        h10 = t3 - 2*t2 + t
        h01 = -2*t3 + 3*t2
        h11 = t3 - t2
        
        # Compute the interpolated value
        y[i] = (h00 * ya[klo] +
                h10 * h * derivatives[klo] +
                h01 * ya[khi] +
                h11 * h * derivatives[khi])
    
    # Return result as scalar if input was scalar
    if scalar:
        return y[0]
    
    return y






def neville_optimized_vec_3(x_points, y_points, X, path = 'closest'):
    
    x_points = np.asarray(x_points, dtype = np.float64)
    y_points = np.asarray(y_points, dtype = np.float64)
    X = np.asarray(X, dtype = np.float64)
    
    scalar = False
    
    if X.ndim == 0:
        X = X[None]
        scalar = True
    
    num_points = len(x_points)
    num_targets = len(X)
    
    F = np.zeros((num_targets,2), dtype = np.float64)
    
    for j in range(num_targets):
        #print('--point ',j+1,' --')
        target_x = X[j]
        D = np.array(y_points.copy(), dtype = np.float64)  # U represents near differences
        U = np.array(y_points.copy(), dtype = np.float64)  # D represents far differences
        nearest_index = 0
        last_diff = 0.0  # Last difference used for error estimation

        # Find the index of the closest point in x_points to target_x
        min_diff = np.abs(target_x - x_points[0])
        for i in range(num_points):
            current_diff = np.abs(target_x - x_points[i])
            if current_diff < min_diff:
                nearest_index = i
                min_diff = current_diff








        interpolated_y = y_points[nearest_index]
        
        if path =='closest':
            use_U = target_x > x_points[nearest_index]
        elif path == 'variant':
            use_U = target_x < x_points[nearest_index]

        if nearest_index == 0:
            use_U = False
    
        if nearest_index == num_points - 1:
            use_U = True

        index = nearest_index

        for order in range(1, num_points):
            
            #print("Max index", num_points - order)

            #Calculation of the D's and U's of the column m
            for i in range(num_points - order):
                x_diff_near = x_points[i] - target_x
                x_diff_far = x_points[i + order] - target_x
                w = D[i + 1] - U[i] #we use the U's and D's 
                denominator = x_diff_near - x_diff_far
                if denominator == 0.0:
                    raise Exception("Zero denominator encountered in neville_interpolation")
                # Update U and D
                denominator = w / denominator
                U[i] = x_diff_far * denominator
                D[i] = x_diff_near * denominator


            if use_U == True:

                index -= 1
                #print("U",index,order)
                last_diff = U[index]

                #Alternate to D
                use_U = False

                #Check if i can go down
                if index > (num_points - order - 1) -1:
                    use_U = True

            else:
                #print("D",index,order)
                last_diff = D[index]

                #Alternate to U
                use_U = True

                #Check if I can go up
                if index - 1 < 0:
                    use_U = False

            # Choose the appropriate difference based on the "straightest line" path
            #if 2 * (nearest_index + 1) < (num_points - order):
                #last_diff = D[nearest_index + 1]
            #else:
                #last_diff = U[nearest_index]
                #nearest_index -= 1
            #print("High index", len(U)-1-order)
            interpolated_y += last_diff
            
        F[j,:] = (interpolated_y,abs(last_diff))

    
    if scalar:
        return F[0]
    
    return F

