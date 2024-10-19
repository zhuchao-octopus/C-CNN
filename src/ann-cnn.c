/////////////////////////////////////////////////////////////////////////////////////////////
/*
 *  ann-cnn.c
 *  Home Page :http://www.1234998.top
 *  Created on: Mar 29, 2023
 *  Author: M
 *
 *  Description:
 *  This file implements the core functionality of a Convolutional Neural Network (CNN).
 *  It includes definitions for various layer types, random number generation utilities,
 *  and mathematical constants used throughout the CNN operations.
 *
 *  Platform Support:
 *  This implementation is designed to be compatible with the STM32 platform.
 *  It utilizes USART for serial communication and includes necessary libraries.
 */
/////////////////////////////////////////////////////////////////////////////////////////////

#include "ann-cnn.h"  // Include the header file containing declarations for CNN functions and structures

#ifdef PLATFORM_STM32  // Conditional compilation for STM32 platform
#include "usart.h"     // Include USART header for serial communication
#include "octopus.h"   // Include additional header files as needed for STM32 functionalities
#endif

// Array of strings representing the names of different CNN layer types
char* CNNTypeName[] =
{
    "Input",          // Input layer
    "Convolution",    // Convolutional layer
    "ReLu",           // Rectified Linear Unit activation layer
    "Pool",           // Pooling layer
    "FullyConnection", // Fully connected (dense) layer
    "SoftMax",        // SoftMax output layer for classification
    "None"           // Placeholder for no layer
};

// Define constants for random number generation
#define RAN_RAND_MAX        2147483647            // Maximum value for random number generation
#define RAN_RNOR_C              1.7155277699214135    // Constant for generating normally distributed random numbers
#define RAN_RNOR_R              (1.0 / RAN_RNOR_C)     // Inverse of the normal distribution constant
#define RAN_IEEE_1                 4.656612877414201e-10  // Precision constant for IEEE floating-point numbers
#define RAN_IEEE_2                 2.220446049250313e-16  // Another precision constant for IEEE floating-point numbers
#define M_PI                           3.14159265358979323846 // Mathematical constant for π (pi)

// Additional function implementations and definitions will follow this section

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
// Function to get the current timestamp in milliseconds
time_t GetTimestamp(void)
{
    time_t t = clock(); // Retrieve processor time used by the program
    time_t tick = t * 1000 / CLOCKS_PER_SEC; // Convert clock ticks to milliseconds
    return tick; // Return the timestamp in milliseconds
    // Optionally, can also return the current calendar time in seconds with: return time(NULL);
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

// Function to generate a Gaussian random number
// This function produces a sequence of Gaussian distributed random numbers
// with a mean (expectation) of μ and a variance of σ² (Variance).
// If a random variable X follows a normal distribution with expectation μ and variance σ²,
// it is denoted as N(μ, σ²).
// The probability density function of the normal distribution is defined by its mean μ,
// which determines its location, and its standard deviation σ, which determines the spread of the distribution.
// The standard normal distribution occurs when μ = 0 and σ = 1.

float32_t NeuralNet_GetGaussRandom(float32_t mul, float32_t Variance)
{
    return mul + GenerateGaussRandom2() * Variance; // Generate a Gaussian random number adjusted by the mean and variance
}

// Function to generate Gaussian noise using the Box-Muller transform
// This function generates Gaussian-distributed random numbers with a specified mean (mu)
// and standard deviation (sigma). It utilizes the Box-Muller transform for generating
// two independent standard normally distributed random numbers from uniformly distributed
// random numbers.

float32_t generateGaussianNoise_Box_Muller(float32_t mu, float32_t sigma)
{
    static const float32_t epsilon = DBL_MIN;  // Smallest positive float value to avoid division by zero
    static const float32_t two_pi = 2.0 * M_PI; // Precomputed value of 2π for later calculations
    static float32_t z0, z1; // Store two generated Gaussian numbers
    static uint32_t generate; // Flag to alternate between generating two values

    generate = !generate; // Toggle the generate flag
    if (!generate)
        return z1 * sigma + mu; // Return the second value if already generated

    float32_t u1, u2; // Variables for uniformly distributed random numbers
    do
    {
        // Generate two uniform random numbers in the range (0, 1)
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    }
    while (u1 <= epsilon); // Ensure u1 is not zero to avoid log(0)

    // Apply the Box-Muller transform to generate two standard normal variables
    float32_t sqrtVal = sqrt(-2.0 * log(u1)); // Calculate the square root component
    z0 = sqrtVal * cos(two_pi * u2); // First standard normal variable
    z1 = sqrtVal * sin(two_pi * u2); // Second standard normal variable

    return z0 * sigma + mu; // Return the Gaussian random number adjusted by mu and sigma
}

// Function to generate Gaussian noise using the Ziggurat algorithm
// This function generates Gaussian-distributed random numbers with a specified mean (mu)
// and standard deviation (sigma). The Ziggurat algorithm is a highly efficient method
// for generating random numbers from a normal distribution.

float32_t generateGaussianNoise_Ziggurat(float32_t mu, float32_t sigma)
{
    static uint32_t iu, iv; // Static variables for indexing
    static float32_t x, y, q; // Variables to hold generated values
    static float32_t u[256], v[256]; // Arrays to store precomputed values for the Ziggurat
    static uint32_t init = 0; // Initialization flag

    // Initialize the Ziggurat algorithm once
    if (!init)
    {
        srand(time(NULL)); // Seed the random number generator

        float32_t c, d, e; // Intermediate variables for calculations
        float32_t f, g; // Variables for final calculations
        float32_t h; // Uniform random variable

        c = RAN_RNOR_C; // Constant for the algorithm
        d = fabs(c); // Absolute value of the constant

        // Precompute the Ziggurat values
        while (1)
        {
            do
            {
                // Generate two uniform random variables
                x = 6.283185307179586476925286766559 * rand() / RAN_RAND_MAX; // Uniformly distributed angle
                y = 1.0 - RAN_IEEE_1 * (h = rand() / RAN_RAND_MAX); // Uniformly distributed height
                q = h * exp(-0.5 * x * x); // Calculate the area under the curve

                // Check against thresholds and compute corresponding values
                if (q <= 0.2891349)
                {
                    // Polynomial approximation for low values
                    e = h * ((((((((((((
                                           -0.000200214257568 * h
                                           + 0.000100950558225) * h
                                       + 0.001349343676503) * h
                                      - 0.003673428679726) * h
                                     + 0.005739507733706) * h
                                    - 0.007622461300117) * h
                                   + 0.009438870047315) * h
                                  + 1.00167406037274) * h
                                 + 2.83297611084620) * h
                                + 1.28067431755817) * h
                               + 0.564189583547755) * h
                              + 9.67580788298760e-1) * h
                             + 1.0);
                    break;
                }

                // Polynomial approximation for medium values
                if (q <= 0.67780119)
                {
                    e = h * (((((((((((((((((
                                                -0.000814322055558 * h
                                                + 0.00027344107956) * h
                                            + 0.00134049846717) * h
                                           - 0.00294315816186) * h
                                          + 0.00481439291198) * h
                                         - 0.00653236492407) * h
                                        + 0.00812419176390) * h
                                       - 0.01003558218763) * h
                                      + 0.01196480954895) * h
                                     - 0.01443119616267) * h
                                    + 0.01752937625694) * h
                                   - 0.02166037955289) * h
                                  + 1.00393389947532) * h
                                 + 2.96952232392818) * h
                                + 1.28067430575358) * h
                               + 0.56418958354775) * h
                              + 9.67580788298599e-1) * h
                             + 1.0);
                    break;
                }

                // Check for acceptance by comparing against the Gaussian tail
                if (x * x + log(h) / d <= -2.0 * log(x))
                {
                    e = h; // Assign accepted height
                    break;
                }
            }
            while (y > q); // Keep looping until a valid sample is found

            // Compute final values to store in the Ziggurat arrays
            f = (x >= 0.0) ? d + x : -d - x; // Shift value based on sign of x
            g = exp(0.5 * f * f); // Exponential component
            u[init] = f * g; // Store computed value in u
            v[init] = g * g; // Store squared value in v

            init++; // Increment the initialization index
            if (init >= 256) // Ensure we do not exceed array bounds
                break; // Exit if arrays are filled
        }
    }

    // Generate the Gaussian noise using precomputed values
    init--; // Decrement the index for retrieving values
    if (init < 0) // Wrap around if the index is negative
        init = 255;

    x = u[init]; // Retrieve the current u value
    y = v[init]; // Retrieve the current v value
    q = (init > 0) ? v[init - 1] : v[255]; // Get the previous v value or the last one

    float32_t result = RAN_RNOR_R * x / q; // Calculate the final Gaussian value

    return sigma * result + mu; // Scale and shift the result to the desired distribution
}

// Function to generate a random floating-point number between 0 and 1
float32_t GenerateRandomNumber()
{
    // Set the random number seed (commented out)
    // This line seeds the random number generator with the current time,
    // which helps ensure different random numbers each time the program runs.
    // srand(time(NULL));

    // Generate a random integer between 0 and RAND_MAX
    uint32_t randInt = rand(); // rand() returns a pseudo-random integer

    // Normalize the random integer to a floating-point number between 0 and 1
    // The result is cast to float32_t to ensure it is a floating-point value.
    float32_t randFloat = (float32_t)randInt / RAND_MAX; // Normalize to range [0, 1]

    // Return the generated random float
    return randFloat; // This returns a number in the range [0, 1]
}

float32_t GenerateGaussRandom(void)
{
    float32_t c, u, v, r; // Declare local variables
    static bool return_v = false; // Static variable to track if the previous value should be returned
    static float32_t v_val; // Store the last generated value

    c = 0; // Initialize c
    if (return_v) // If we need to return the previously generated value
    {
        return_v = false; // Reset the flag
        return v_val; // Return the last generated value
    }

#ifdef PLATFORM_STM32
    // Generate two random numbers on STM32 platform
    u = 2 * random() - 1; // Generate a random number in the range [-1, 1]
    v = 2 * random() - 1;
#else
    // Generate random numbers on other platforms
    u = 2 * rand() - 1; // Generate a random number in the range [-1, 1]
    v = 2 * rand() - 1;
#endif

    r = u * u + v * v; // Calculate r
    if ((r == 0) || (r > 1)) // If r is 0 or greater than 1, regenerate the Gaussian random number
    {
        return GenerateGaussRandom(); // Recursive call
    }

#ifdef PLATFORM_STM32
    // Calculate c on STM32 platform
    arm_sqrt_f32(-2 * log10(r) / r, &c);
#else
    // Calculate c on other platforms
    c = sqrt(-2 * log10(r) / r);
#endif

    v_val = v * c; // Calculate v's value
    return_v = true; // Set the flag to indicate a value will be returned
    return u * c; // Return the Gaussian random number
}

float32_t GenerateGaussRandom1(void)
{
    static float32_t v1, v2, s; // Static variables to store intermediate values
    static uint32_t start = 0; // Flag to alternate between two generated values
    float32_t x; // Variable to store the generated Gaussian random number

    if (start == 0) // If this is the first call
    {
        do
        {
            // Generate two uniform random numbers in the range [0, 1]
            float32_t u1 = (float32_t)rand() / RAND_MAX; // First uniform random number
            float32_t u2 = (float32_t)rand() / RAND_MAX; // Second uniform random number

            // Transform uniform random numbers to [-1, 1]
            v1 = 2 * u1 - 1;
            v2 = 2 * u2 - 1;

            // Calculate the sum of squares
            s = v1 * v1 + v2 * v2;
        }
        while (s >= 1 || s == 0); // Continue until we have a valid s

        // Calculate the Gaussian random number using Box-Muller transform
        x = v1 * sqrt(-2 * log(s) / s);
    }
    else // If it's the second call
    {
        // Use the second value to generate a new Gaussian random number
        x = v2 * sqrt(-2 * log(s) / s);
    }

    start = 1 - start; // Toggle the start flag to alternate between values
    return x; // Return the generated Gaussian random number
}

float32_t GenerateGaussRandom2(void)
{
    static float32_t n2 = 0.0; // Cached second value of the generated Gaussian random number
    static uint32_t n2_cached = 0; // Flag to indicate if n2 is cached
    float32_t d; // Variable to store the calculated distance for Box-Muller transform

    if (!n2_cached) // If n2 is not cached
    {
        float32_t x, y, r; // Variables to hold generated points and radius
        do
        {
            // Generate two uniform random numbers in the range [-1, 1]
            x = 2.0 * rand() / RAND_MAX - 1; // First uniform random number
            y = 2.0 * rand() / RAND_MAX - 1; // Second uniform random number

            // Calculate the radius squared
            r = x * x + y * y;
        }
        while (r >= 1.0 || r == 0.0); // Continue until we have a valid radius

        // Calculate the distance using Box-Muller transform
#ifdef PLATFORM_STM32
        arm_sqrt_f32(-2 * log10(r) / r, &d); // Use ARM-specific function for square root
#else
        d = sqrt(-2.0 * log(r) / r); // Standard square root calculation
#endif

        float32_t n1 = x * d; // First Gaussian random number
        n2 = y * d; // Second Gaussian random number (cached for next call)
        n2_cached = 1; // Mark that n2 has been cached
        return n1; // Return the first Gaussian random number
    }
    else // If n2 is cached
    {
        n2_cached = 0; // Reset the cache flag
        return n2; // Return the cached second Gaussian random number
    }
}

// Function to determine the sign of a number
int sign(float32_t x)
{
    return (x >= 0) ? 1 : -1; // Return 1 for non-negative, -1 for negative
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Creates a tensor entity and allocates space for the tensor.
/// </summary>
/// <param name="Length">The length of the tensor to be created.</param>
/// <returns>A pointer to the created tensor, or NULL if allocation fails.</returns>
TPTensor MakeTensor(uint32_t Length)
{
    // Allocate memory for the tensor structure
    TPTensor tPTensor = malloc(sizeof(TTensor));
    if (tPTensor != NULL) // Check if memory allocation was successful
    {
        tPTensor->length = Length; // Set the length of the tensor
        // Allocate memory for the tensor's buffer
        tPTensor->buffer = malloc(tPTensor->length * sizeof(float32_t));
        if (tPTensor->buffer != NULL) // Check if buffer allocation was successful
        {
            // Initialize the buffer to zero
            memset(tPTensor->buffer, 0, tPTensor->length * sizeof(float32_t));
        }
    }
    return tPTensor; // Return the created tensor
}

/////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>
/// Initializes a tensor entity by allocating space and setting its dimensions.
/// </summary>
/// <param name="PTensor">Pointer to the tensor to initialize.</param>
/// <param name="W">Width of the tensor.</param>
/// <param name="H">Height of the tensor.</param>
/// <param name="Depth">Depth of the tensor.</param>
/// <param name="Bias">Bias value used for initialization.</param>
void TensorInit(TPTensor PTensor, uint16_t W, uint16_t H, uint16_t Depth, float32_t Bias)
{
    uint32_t n = W * H * Depth; // Calculate the total number of elements in the tensor
    PTensor = MakeTensor(n); // Allocate memory for the tensor
    TensorFillZero(PTensor); // Fill the tensor with zeros
}

/// <summary>
/// Fills the tensor's buffer with zeros.
/// </summary>
/// <param name="PTensor">Pointer to the tensor to fill with zeros.</param>
void TensorFillZero(TPTensor PTensor)
{
    if (PTensor->length > 0) // Check if the tensor has a valid length
        memset(PTensor->buffer, 0, PTensor->length * sizeof(float32_t)); // Set all elements to zero
}

/// <summary>
/// Fills the tensor's buffer with a specified bias value.
/// </summary>
/// <param name="PTensor">Pointer to the tensor to fill.</param>
/// <param name="Bias">Value to fill the tensor with.</param>
void TensorFillWith(TPTensor PTensor, float32_t Bias)
{
    for (int i = 0; i < PTensor->length; i++)
    {
        PTensor->buffer[i] = Bias; // Assign the bias value to each element
    }
}

/// <summary>
/// Fills the tensor's buffer with Gaussian random values.
/// </summary>
/// <param name="PTensor">Pointer to the tensor to fill with Gaussian noise.</param>
void TensorFillGauss(TPTensor PTensor)
{
    float32_t scale = 0; // Variable to store the scaling factor
    // Calculate the scaling factor based on the tensor's length
#ifdef PLATFORM_STM32
    arm_sqrt_f32(1.0 / PTensor->length, &scale); // Calculate scale for STM32 platform
#else
    scale = sqrt(1.0 / PTensor->length); // Calculate scale for other platforms
#endif

    for (int i = 0; i < PTensor->length; i++)
    {
        float32_t v = NeuralNet_GetGaussRandom(0.00, scale); // Get a Gaussian random value
        PTensor->buffer[i] = v; // Assign the random value to the tensor
    }
}

/// <summary>
/// Computes the maximum and minimum values from the tensor.
/// </summary>
/// <param name="PTensor">Pointer to the tensor from which to find max and min values.</param>
/// <returns>Pointer to a structure containing the max and min values.</returns>
TPMaxMin TensorMaxMin(TPTensor PTensor)
{
    TPMaxMin pMaxMin = malloc(sizeof(TMaxMin)); // Allocate memory for max/min structure
    pMaxMin->max = MINFLOAT_NEGATIVE_NUMBER; // Initialize max to the smallest possible value
    pMaxMin->min = MAXFLOAT_POSITIVE_NUMBER; // Initialize min to the largest possible value

    // Iterate through the tensor's buffer to find max and min values
    for (int i = 0; i < PTensor->length; i++)
    {
        if (PTensor->buffer[i] > pMaxMin->max)
            pMaxMin->max = PTensor->buffer[i]; // Update max if the current value is greater
        if (PTensor->buffer[i] < pMaxMin->min)
            pMaxMin->min = PTensor->buffer[i]; // Update min if the current value is smaller
    }
    return pMaxMin; // Return the pointer to the max/min structure
}

/// <summary>
/// Frees the memory allocated for the tensor and its buffer.
/// </summary>
/// <param name="PTensor">Pointer to the tensor to free.</param>
void TensorFree(TPTensor PTensor)
{
    if (PTensor != NULL) // Check if the tensor is not NULL
    {
        free(PTensor->buffer); // Free the buffer memory
        PTensor->length = 0; // Reset length to 0
        free(PTensor); // Free the tensor structure
        PTensor = NULL; // Set pointer to NULL to avoid dangling reference
    }
}

/// <summary>
/// Saves the tensor's buffer data to a file.
/// </summary>
/// <param name="pFile">File pointer to write the tensor data to.</param>
/// <param name="PTensor">Pointer to the tensor to save.</param>
void TensorSave(FILE* pFile, TPTensor PTensor)
{
    if (pFile == NULL) // Check if the file pointer is NULL
    {
        LOG("Error opening file NULL"); // Log an error message
        return; // Exit the function
    }
    // Write the tensor's buffer to the file if it's valid
    if (PTensor->buffer != NULL && PTensor->length > 0)
        fwrite(PTensor->buffer, 1, sizeof(float32_t) * PTensor->length, pFile);
}

/// <summary>
/// Loads tensor data from a file into the tensor's buffer.
/// </summary>
/// <param name="pFile">File pointer to read the tensor data from.</param>
/// <param name="PTensor">Pointer to the tensor to load data into.</param>
void TensorLoad(FILE* pFile, TPTensor PTensor)
{
    if (pFile == NULL) // Check if the file pointer is NULL
    {
        LOG("Error opening file NULL"); // Log an error message
        return; // Exit the function
    }
    // Read data from the file into the tensor's buffer if it's valid
    if (PTensor->buffer != NULL && PTensor->length > 0)
        fread(PTensor->buffer, 1, sizeof(float32_t) * PTensor->length, pFile);
}

/////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////

/// <summary>
/// Creates a volume and allocates memory for it.
/// </summary>
/// <param name="W">Width of the volume.</param>
/// <param name="H">Height of the volume.</param>
/// <param name="Depth">Depth of the volume.</param>
/// <returns>Pointer to the created volume.</returns>
TPVolume MakeVolume(uint16_t W, uint16_t H, uint16_t Depth)
{
    TPVolume tPVolume = malloc(sizeof(TVolume)); // Allocate memory for the volume
    if (tPVolume != NULL) // Check if memory allocation was successful
    {
        // Set the dimensions of the volume
        tPVolume->_w = W;
        tPVolume->_h = H;
        tPVolume->_depth = Depth;

        // Initialize function pointers for volume operations
        tPVolume->init = VolumeInit;
        tPVolume->free = TensorFree; // Reuse TensorFree for memory cleanup
        tPVolume->fillZero = TensorFillZero; // Set all values to zero
        tPVolume->fillGauss = TensorFillGauss; // Fill with Gaussian random values
        tPVolume->setValue = VolumeSetValue; // Set specific values in the volume
        tPVolume->getValue = VolumeGetValue; // Get specific values from the volume
        tPVolume->addValue = VolumeAddValue; // Add values to the volume
        tPVolume->setGradValue = VolumeSetGradValue; // Set gradient values
        tPVolume->getGradValue = VolumeGetGradValue; // Get gradient values
        tPVolume->addGradValue = VolumeAddGradValue; // Add gradient values
        tPVolume->flip = VolumeFlip; // Flip the volume
        tPVolume->print = VolumePrint; // Print volume details
    }
    return tPVolume; // Return the pointer to the created volume
}

/// <summary>
/// Initializes the volume with given dimensions and bias value.
/// </summary>
/// <param name="PVolume">Pointer to the volume to initialize.</param>
/// <param name="W">Width of the volume.</param>
/// <param name="H">Height of the volume.</param>
/// <param name="Depth">Depth of the volume.</param>
/// <param name="Bias">Bias value to initialize the volume's tensors.</param>
void VolumeInit(TPVolume PVolume, uint16_t W, uint16_t H, uint16_t Depth, float32_t Bias)
{
    PVolume->_w = W; // Set the width
    PVolume->_h = H; // Set the height
    PVolume->_depth = Depth; // Set the depth

    uint32_t n = PVolume->_w * PVolume->_h * PVolume->_depth; // Calculate total size
    PVolume->weight = MakeTensor(n); // Create tensor for weights
    PVolume->grads = MakeTensor(n); // Create tensor for gradients

    // Initialize the weight and gradient tensors with the bias value
    TensorFillWith(PVolume->weight, Bias);
    TensorFillWith(PVolume->grads, Bias);
}

/// <summary>
/// Frees the allocated memory for the volume and its tensors.
/// </summary>
/// <param name="PVolume">Pointer to the volume to free.</param>
void VolumeFree(TPVolume PVolume)
{
    if (PVolume != NULL) // Check if the volume is not NULL
    {
        TensorFree(PVolume->weight); // Free the weights tensor
        TensorFree(PVolume->grads); // Free the gradients tensor
        free(PVolume); // Free the volume structure
    }
    PVolume = NULL; // Set pointer to NULL to avoid dangling reference
}

/// <summary>
/// Sets the value at the specified coordinates in the volume's weight tensor.
/// </summary>
/// <param name="PVolume">Pointer to the volume.</param>
/// <param name="X">X coordinate.</param>
/// <param name="Y">Y coordinate.</param>
/// <param name="Depth">Depth index.</param>
/// <param name="Value">Value to set.</param>
void VolumeSetValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value)
{
    // Calculate the linear index for the 3D coordinates in the 1D buffer
    uint32_t index = ((PVolume->_w * Y) + X) * PVolume->_depth + Depth;
    PVolume->weight->buffer[index] = Value; // Set the value at the calculated index
}

/// <summary>
/// Adds a value to the existing value at the specified coordinates in the volume's weight tensor.
/// </summary>
/// <param name="PVolume">Pointer to the volume.</param>
/// <param name="X">X coordinate.</param>
/// <param name="Y">Y coordinate.</param>
/// <param name="Depth">Depth index.</param>
/// <param name="Value">Value to add.</param>
void VolumeAddValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value)
{
    // Calculate the linear index for the 3D coordinates in the 1D buffer
    uint32_t index = ((PVolume->_w * Y) + X) * PVolume->_depth + Depth;
    PVolume->weight->buffer[index] = PVolume->weight->buffer[index] + Value; // Add the value to the existing value
}

/// <summary>
/// Retrieves the value at the specified coordinates in the volume's weight tensor.
/// </summary>
/// <param name="PVolume">Pointer to the volume.</param>
/// <param name="X">X coordinate.</param>
/// <param name="Y">Y coordinate.</param>
/// <param name="Depth">Depth index.</param>
/// <returns>The value at the specified coordinates.</returns>
float32_t VolumeGetValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth)
{
    // Calculate the linear index for the 3D coordinates in the 1D buffer
    uint32_t index = ((PVolume->_w * Y) + X) * PVolume->_depth + Depth;
    return PVolume->weight->buffer[index]; // Return the value at the calculated index
}

/// <summary>
/// Sets the gradient value at the specified coordinates in the volume's gradient tensor.
/// </summary>
/// <param name="PVolume">Pointer to the volume.</param>
/// <param name="X">X coordinate.</param>
/// <param name="Y">Y coordinate.</param>
/// <param name="Depth">Depth index.</param>
/// <param name="Value">Gradient value to set.</param>
void VolumeSetGradValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value)
{
    // Calculate the linear index for the 3D coordinates in the 1D buffer
    uint32_t index = ((PVolume->_w * Y) + X) * PVolume->_depth + Depth;
    PVolume->grads->buffer[index] = Value; // Set the gradient value at the calculated index
}

/// <summary>
/// Adds a gradient value to the existing gradient value at the specified coordinates in the volume's gradient tensor.
/// </summary>
/// <param name="PVolume">Pointer to the volume.</param>
/// <param name="X">X coordinate.</param>
/// <param name="Y">Y coordinate.</param>
/// <param name="Depth">Depth index.</param>
/// <param name="Value">Gradient value to add.</param>
void VolumeAddGradValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth, float32_t Value)
{
    // Calculate the linear index for the 3D coordinates in the 1D buffer
    uint32_t index = ((PVolume->_w * Y) + X) * PVolume->_depth + Depth;
    PVolume->grads->buffer[index] = PVolume->grads->buffer[index] + Value; // Add the gradient value to the existing gradient value
}

/// <summary>
/// Retrieves the gradient value at the specified coordinates in the volume's gradient tensor.
/// </summary>
/// <param name="PVolume">Pointer to the volume.</param>
/// <param name="X">X coordinate.</param>
/// <param name="Y">Y coordinate.</param>
/// <param name="Depth">Depth index.</param>
/// <returns>The gradient value at the specified coordinates.</returns>
float32_t VolumeGetGradValue(TPVolume PVolume, uint16_t X, uint16_t Y, uint16_t Depth)
{
    // Calculate the linear index for the 3D coordinates in the 1D buffer
    uint32_t index = ((PVolume->_w * Y) + X) * PVolume->_depth + Depth;
    return PVolume->grads->buffer[index]; // Return the gradient value at the calculated index
}

/// <summary>
/// Flips the volume along the width dimension for each depth channel.
/// </summary>
/// <param name="PVolume">Pointer to the volume.</param>
void VolumeFlip(TPVolume PVolume)
{
    uint16_t width = PVolume->_w;  // Width of the volume
    uint16_t height = PVolume->_h; // Height of the volume
    uint16_t depth = PVolume->_depth; // Depth of the volume

    // Flip each channel
    for (uint16_t d = 0; d < depth; d++)
    {
        for (uint16_t y = 0; y < height; y++)
        {
            for (uint16_t x = 0; x < width / 2; x++)
            {
                // Swap pixel positions to flip the volume
                float32_t temp = PVolume->weight->buffer[y * width * depth + x * depth + d];
                PVolume->weight->buffer[y * width * depth + x * depth + d] = PVolume->weight->buffer[y * width * depth + (width - x - 1) * depth + d];
                PVolume->weight->buffer[y * width * depth + (width - x - 1) * depth + d] = temp;
            }
        }
    }
}

/// <summary>
/// Creates a set of filters and allocates memory for them.
/// </summary>
/// <param name="W">Width of each filter.</param>
/// <param name="H">Height of each filter.</param>
/// <param name="Depth">Depth of each filter.</param>
/// <param name="FilterNumber">Number of filters to create.</param>
/// <returns>A pointer to the created filters.</returns>
TPFilters MakeFilters(uint16_t W, uint16_t H, uint16_t Depth, uint16_t FilterNumber)
{
    TPFilters tPFilters = malloc(sizeof(TFilters)); // Allocate memory for the filters
    if (tPFilters != NULL)
    {
        tPFilters->_w = W; // Set width
        tPFilters->_h = H; // Set height
        tPFilters->_depth = Depth; // Set depth
        tPFilters->filterNumber = FilterNumber; // Set the number of filters
        tPFilters->volumes = malloc(sizeof(TPVolume) * tPFilters->filterNumber); // Allocate memory for filter volumes
        if (tPFilters->volumes == NULL)
        {
            LOGERROR("tPFilters->volumes==NULL! W=%d H=%d Depth=%d FilterNumber=%d", W, H, Depth, FilterNumber);
            return NULL; // Return NULL if allocation fails
        }
        for (uint16_t i = 0; i < tPFilters->filterNumber; i++)
        {
            tPFilters->volumes[i] = MakeVolume(tPFilters->_w, tPFilters->_h, tPFilters->_depth); // Create each volume
        }
        tPFilters->init = VolumeInit; // Assign initialization function
        tPFilters->free = FilterVolumesFree; // Assign free function
    }
    return tPFilters; // Return the created filters
}

/// <summary>
/// Resizes the filter set with new dimensions and number of filters.
/// </summary>
/// <param name="PFilters">Pointer to the filters.</param>
/// <param name="W">New width for filters.</param>
/// <param name="H">New height for filters.</param>
/// <param name="Depth">New depth for filters.</param>
/// <param name="FilterNumber">New number of filters.</param>
/// <returns>True if resizing is successful, false otherwise.</returns>
bool FiltersResize(TPFilters PFilters, uint16_t W, uint16_t H, uint16_t Depth, uint16_t FilterNumber)
{
    // Validate the new dimensions
    if (W <= 0 || H <= 0 || Depth <= 0 || FilterNumber <= 0)
    {
        LOGERROR("Resize Filters failed! W=%d H=%d Depth=%d FilterNumber=%d", W, H, Depth, FilterNumber);
        return false; // Return false if invalid dimensions
    }

    if (PFilters != NULL)
    {
        PFilters->_w = W; // Update width
        PFilters->_h = H; // Update height
        PFilters->_depth = Depth; // Update depth
        PFilters->filterNumber = FilterNumber; // Update the number of filters
        PFilters->free(PFilters); // Free existing volumes
        PFilters->volumes = malloc(sizeof(TPVolume) * PFilters->filterNumber); // Allocate memory for new filter volumes
        for (uint16_t i = 0; i < PFilters->filterNumber; i++)
        {
            PFilters->volumes[i] = MakeVolume(PFilters->_w, PFilters->_h, PFilters->_depth); // Create new volumes
        }
        PFilters->init = VolumeInit; // Reassign initialization function
        PFilters->free = FilterVolumesFree; // Reassign free function
    }
    return true; // Return true if resizing is successful
}

/// <summary>
/// Frees the allocated memory for the filter set and its volumes.
/// </summary>
/// <param name="PFilters">Pointer to the filters.</param>
void FiltersFree(TPFilters PFilters)
{
    if (PFilters != NULL)
    {
        // Free each volume in the filter set
        for (uint16_t d = 0; d < PFilters->filterNumber; d++)
        {
            VolumeFree(PFilters->volumes[d]);
        }
        free(PFilters); // Free the filter set
        PFilters = NULL; // Set pointer to NULL
    }
}

/// <summary>
/// Frees the memory allocated for the volumes in the filter set.
/// </summary>
/// <param name="PFilters">Pointer to the filters.</param>
void FilterVolumesFree(TPFilters PFilters)
{
    if (PFilters != NULL)
    {
        // Free each volume without freeing the filter set itself
        for (uint16_t d = 0; d < PFilters->filterNumber; d++)
        {
            VolumeFree(PFilters->volumes[d]);
        }
        PFilters->volumes = NULL; // Set volume pointer to NULL
    }
}

/// @brief Prints the values or gradients of a volume.
/// @param PVolume Pointer to the volume to print.
/// @param wg 0 for weights, 1 for gradients.
void VolumePrint(TPVolume PVolume, uint8_t wg)
{
    // Check if the volume is 1x1 in dimensions
    if (PVolume->_h == 1 && PVolume->_w == 1)
    {
        if (wg == PRINTFLAG_WEIGHT)
            LOGINFOR("weight:PVolume->_depth=%d/%d", PVolume->_depth, PVolume->_depth);
        else
            LOGINFOR("grads:PVolume->_depth=%d/%d", PVolume->_depth, PVolume->_depth);
    }

    float32_t f32 = 0.00; // Initialize value variable
    for (uint16_t d = 0; d < PVolume->_depth; d++)
    {
        if (PVolume->_h == 1 && PVolume->_w == 1)
        {
            // Print individual depth values
            if (wg == PRINTFLAG_WEIGHT)
            {
                f32 = PVolume->getValue(PVolume, 0, 0, d);
                (d == 0) ? LOG(PRINTFLAG_FORMAT, f32) : LOG("," PRINTFLAG_FORMAT, f32);
            }
            else
            {
                f32 = PVolume->getGradValue(PVolume, 0, 0, d);
                (d == 0) ? LOG(PRINTFLAG_FORMAT, f32) : LOG("," PRINTFLAG_FORMAT, f32);
            }
        }
        else
        {
            // Print values/gradients for non-1x1 dimensions
            if (wg == PRINTFLAG_WEIGHT)
                LOGINFOR("weight:PVolume->_depth=%d/%d %dx%d", d, PVolume->_depth, PVolume->_w, PVolume->_h);
            else
                LOGINFOR("grads:PVolume->_depth=%d/%d %dx%d", d, PVolume->_depth, PVolume->_w, PVolume->_h);

            for (uint16_t y = 0; y < PVolume->_h; y++)
            {
                for (uint16_t x = 0; x < PVolume->_w; x++)
                {
                    if (wg == PRINTFLAG_WEIGHT)
                    {
                        f32 = PVolume->getValue(PVolume, x, y, d);
                        (x == 0) ? LOG(PRINTFLAG_FORMAT, f32) : LOG("," PRINTFLAG_FORMAT, f32);
                    }
                    else
                    {
                        f32 = PVolume->getGradValue(PVolume, x, y, d);
                        (x == 0) ? LOG(PRINTFLAG_FORMAT, f32) : LOG("," PRINTFLAG_FORMAT, f32);
                    }
                }
                LOG("\n"); // Newline after each row
            }
        }
    }
    if (PVolume->_h == 1 && PVolume->_w == 1)
        LOG("\n"); // Newline for 1x1 volume
}

////////////////////////////////////////////////////////////////////////////////////////

/// @brief Initializes the input layer with specified options.
/// @param PInputLayer Pointer to the input layer.
/// @param PLayerOption Pointer to the layer options.
void InputLayerInit(TPInputLayer PInputLayer, TPLayerOption PLayerOption)
{
    PInputLayer->layer.LayerType = PLayerOption->LayerType; // Set layer type
    PInputLayer->layer.in_v = NULL; // Initialize input volume to NULL
    PInputLayer->layer.out_v = NULL; // Initialize output volume to NULL

    // Set input dimensions
    PInputLayer->layer.in_w = PLayerOption->in_w;
    PInputLayer->layer.in_h = PLayerOption->in_h;
    PInputLayer->layer.in_depth = PLayerOption->in_depth;

    // Set output dimensions equal to input dimensions
    PInputLayer->layer.out_w = PLayerOption->out_w = PInputLayer->layer.in_w;
    PInputLayer->layer.out_h = PLayerOption->out_h = PInputLayer->layer.in_h;
    PInputLayer->layer.out_depth = PLayerOption->out_depth = PInputLayer->layer.in_depth;
}

/// @brief Performs the forward pass for the input layer.
/// @param PInputLayer Pointer to the input layer.
/// @param PVolume Pointer to the input volume.
void InputLayerForward(TPInputLayer PInputLayer, TPVolume PVolume)
{
    if (PVolume == NULL)
    {
        LOGERROR("%s is null", CNNTypeName[PInputLayer->layer.LayerType]);
        return; // Exit if input volume is NULL
    }

    // Free previous input volume if it exists
    if (PInputLayer->layer.in_v != NULL)
    {
        VolumeFree(PInputLayer->layer.in_v);
    }

    // Update input dimensions
    PInputLayer->layer.in_w = PVolume->_w;
    PInputLayer->layer.in_h = PVolume->_h;
    PInputLayer->layer.in_depth = PVolume->_depth;
    PInputLayer->layer.in_v = PVolume; // Set the new input volume

    // Set output dimensions equal to input dimensions
    PInputLayer->layer.out_w = PInputLayer->layer.in_w;
    PInputLayer->layer.out_h = PInputLayer->layer.in_h;
    PInputLayer->layer.out_depth = PInputLayer->layer.in_depth;
    PInputLayer->layer.out_v = PInputLayer->layer.in_v; // Output is the same as input
}

/// @brief Placeholder function for the backward pass of the input layer.
void InputLayerBackward(TPInputLayer PInputLayer)
{
    // Placeholder for future implementation
}

/// @brief Frees the resources allocated for the input layer.
/// @param PInputLayer Pointer to the input layer.
void InputLayerFree(TPInputLayer PInputLayer)
{
    VolumeFree(PInputLayer->layer.in_v); // Free input volume
    VolumeFree(PInputLayer->layer.out_v); // Free output volume
}

/// @brief Placeholder for matrix multiplication function.
/// @param PInTenser Pointer to the input tensor.
/// @param PFilter Pointer to the filter tensor.
/// @param out Pointer to the output tensor.
void MatrixMultip(TPTensor PInTenser, TPTensor PFilter, TPTensor out)
{
    // Placeholder for future implementation
}

/////////////////////////////////////////////////////////////////////////////
// @brief 初始化卷积层
// @param PConvLayer - 卷积层指针
// @param PLayerOption - 层选项指针，包含初始化参数
void ConvolutionLayerInit(TPConvLayer PConvLayer, TPLayerOption PLayerOption)
{
    // 设置层类型及基础参数
    PConvLayer->layer.LayerType = PLayerOption->LayerType;
    PConvLayer->l1_decay_rate = PLayerOption->l1_decay_rate;
    PConvLayer->l2_decay_rate = PLayerOption->l2_decay_rate;
    PConvLayer->stride = PLayerOption->stride;
    PConvLayer->padding = PLayerOption->padding;
    PConvLayer->bias = PLayerOption->bias;

    // 设置输入的宽、高和深度（通道数）
    PConvLayer->layer.in_w = PLayerOption->in_w;
    PConvLayer->layer.in_h = PLayerOption->in_h;
    PConvLayer->layer.in_depth = PLayerOption->in_depth;
    PConvLayer->layer.in_v = NULL;

    // 处理滤波器的深度和默认值
    if (PLayerOption->filter_depth != PConvLayer->layer.in_depth)
        PLayerOption->filter_depth = PConvLayer->layer.in_depth;
    if (PLayerOption->filter_depth <= 0)
        PLayerOption->filter_depth = 3;

    // 创建滤波器
    PConvLayer->filters = MakeFilters(PLayerOption->filter_w, PLayerOption->filter_h, PLayerOption->filter_depth, PLayerOption->filter_number);

    // 计算输出的宽度、高度和深度
    PConvLayer->layer.out_w = floor((PConvLayer->layer.in_w + PConvLayer->padding * 2 - PConvLayer->filters->_w) / PConvLayer->stride + 1);
    PConvLayer->layer.out_h = floor((PConvLayer->layer.in_h + PConvLayer->padding * 2 - PConvLayer->filters->_w) / PConvLayer->stride + 1);
    PConvLayer->layer.out_depth = PConvLayer->filters->filterNumber;

    // 创建输出体积并初始化
    PConvLayer->layer.out_v = MakeVolume(PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth);
    PConvLayer->layer.out_v->init(PConvLayer->layer.out_v, PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth, 0);

    // 初始化偏置
    PConvLayer->biases = MakeVolume(1, 1, PConvLayer->layer.out_depth);
    PConvLayer->biases->init(PConvLayer->biases, 1, 1, PConvLayer->layer.out_depth, PConvLayer->bias);

    // 初始化每个滤波器，并使用高斯分布填充权重
    for (uint16_t i = 0; i < PConvLayer->layer.out_depth; i++)
    {
        PConvLayer->filters->init(PConvLayer->filters->volumes[i], PConvLayer->filters->_w, PConvLayer->filters->_h, PConvLayer->filters->_depth, 0);
        PConvLayer->filters->volumes[i]->fillGauss(PConvLayer->filters->volumes[i]->weight);
    }

    // 更新输出的宽度、高度和深度
    PLayerOption->out_w = PConvLayer->layer.out_w;
    PLayerOption->out_h = PConvLayer->layer.out_h;
    PLayerOption->out_depth = PConvLayer->layer.out_depth;
}

/////////////////////////////////////////////////////////////////////////////
// @brief 重新计算卷积层的输出大小
// @param PConvLayer - 卷积层指针
void convolutionLayerOutResize(TPConvLayer PConvLayer)
{
    // 计算新的输出宽度、高度和滤波器深度
    uint16_t out_w = floor((PConvLayer->layer.in_w + PConvLayer->padding * 2 - PConvLayer->filters->_w) / PConvLayer->stride + 1);
    uint16_t out_h = floor((PConvLayer->layer.in_h + PConvLayer->padding * 2 - PConvLayer->filters->_w) / PConvLayer->stride + 1);
    uint16_t filter_depth = PConvLayer->layer.in_depth;

    // 如果滤波器的深度变化，重新调整滤波器
    if (PConvLayer->filters->_depth != filter_depth)
    {
        LOGINFOR("ConvLayer resize filters from %d x %d x%d to %d x %d x %d", PConvLayer->filters->_w, PConvLayer->filters->_h, PConvLayer->filters->_depth, out_w, out_h, filter_depth);
        FiltersFree(PConvLayer->filters);
        bool ret = FiltersResize(PConvLayer->filters, PConvLayer->filters->_w, PConvLayer->filters->_h, filter_depth, PConvLayer->filters->filterNumber);
        if (!ret)
            LOGERROR("Resize Filters failed! W=%d H=%d Depth=%d FilterNumber=%d", PConvLayer->filters->_w, PConvLayer->filters->_h, filter_depth, PConvLayer->filters->filterNumber);
    }

    // 如果输出尺寸发生变化，重新初始化输出体积
    if (PConvLayer->layer.out_w != out_w || PConvLayer->layer.out_h != out_h)
    {
        LOGINFOR("ConvLayer resize out_v from %d x %d to %d x %d", PConvLayer->layer.out_w, PConvLayer->layer.out_h, out_w, out_h);
        PConvLayer->layer.out_w = out_w;
        PConvLayer->layer.out_h = out_h;

        if (PConvLayer->layer.out_v != NULL)
        {
            VolumeFree(PConvLayer->layer.out_v);
        }

        // 创建新的输出体积并初始化
        PConvLayer->layer.out_v = MakeVolume(PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth);
        PConvLayer->layer.out_v->init(PConvLayer->layer.out_v, PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth, 0);
    }
}

/////////////////////////////////////////////////////////////////////////////
// @brief 卷积层的前向运算
// @param PConvLayer - 卷积层指针
// 该函数实现了多通道卷积运算，将输入卷积为输出
void ConvolutionLayerForward(TPConvLayer PConvLayer)
{
    uint16_t padding_x = -PConvLayer->padding;
    uint16_t padding_y = -PConvLayer->padding;
    TPVolume inVolume = PConvLayer->layer.in_v;
    TPVolume outVolume = PConvLayer->layer.out_v;
    uint16_t in_x, in_y;
    float32_t sum = 0.00;

    // 调整输出尺寸
    convolutionLayerOutResize(PConvLayer);

    // 遍历输出的每个深度
    for (uint32_t out_d = 0; out_d < PConvLayer->layer.out_depth; out_d++)
    {
        TPVolume filter = PConvLayer->filters->volumes[out_d];
        padding_x = -PConvLayer->padding;
        padding_y = -PConvLayer->padding;

        // 遍历输出的每个像素
        for (uint32_t out_y = 0; out_y < PConvLayer->layer.out_h; out_y++)
        {
            padding_x = -PConvLayer->padding;
            for (uint32_t out_x = 0; out_x < PConvLayer->layer.out_w; out_x++)
            {
                sum = 0.00;

                // 进行滤波器的卷积操作
                for (uint32_t filter_y = 0; filter_y < PConvLayer->filters->_h; filter_y++)
                {
                    in_y = filter_y + padding_y;
                    if (in_y < 0 || in_y >= inVolume->_h)
                        continue;

                    for (uint32_t filter_x = 0; filter_x < PConvLayer->filters->_w; filter_x++)
                    {
                        in_x = filter_x + padding_x;
                        if (in_x < 0 || in_x >= inVolume->_w)
                            continue;

                        uint32_t fx = (filter->_w * filter_y + filter_x) * filter->_depth;
                        uint32_t ix = (inVolume->_w * in_y + in_x) * inVolume->_depth;

                        // 遍历滤波器的每个深度（多通道卷积）
                        for (uint32_t filter_d = 0; filter_d < PConvLayer->filters->_depth; filter_d++)
                        {
                            sum += filter->weight->buffer[fx + filter_d] * inVolume->weight->buffer[ix + filter_d];
                        }
                    }
                }

                // 加上偏置
                sum += PConvLayer->biases->weight->buffer[out_d];

                // 将卷积结果存入输出体积
                uint32_t out_idx = (outVolume->_w * out_y + out_x) * outVolume->_depth + out_d;
                outVolume->weight->buffer[out_idx] = sum;
                padding_x += PConvLayer->stride;
            }
            padding_y += PConvLayer->stride;
        }
    }
}

/////////////////////////////////////////////////////////////////////////////
// @brief 初始化深度卷积和逐点卷积层
// @param PConvLayer - 卷积层指针
// @param PLayerOption - 层选项指针，包含初始化参数
void DepthwisePointwiseConvolutionInit(TPConvLayer PConvLayer, TPLayerOption PLayerOption)
{
    // 初始化基础参数
    PConvLayer->layer.LayerType = PLayerOption->LayerType;
    PConvLayer->l1_decay_rate = PLayerOption->l1_decay_rate;
    PConvLayer->l2_decay_rate = PLayerOption->l2_decay_rate;
    PConvLayer->stride = PLayerOption->stride;
    PConvLayer->padding = PLayerOption->padding;
    PConvLayer->bias = PLayerOption->bias;

    // 设置输入的宽度、高度和深度
    PConvLayer->layer.in_w = PLayerOption->in_w;
    PConvLayer->layer.in_h = PLayerOption->in_h;
    PConvLayer->layer.in_depth = PLayerOption->in_depth; // 输入的深度等于滤波器的深度
    PConvLayer->layer.in_v = NULL;

    // 确保滤波器深度与输入深度一致
    if (PLayerOption->filter_depth != PConvLayer->layer.in_depth)
        PLayerOption->filter_depth = PConvLayer->layer.in_depth;
    if (PLayerOption->filter_depth <= 0)
        PLayerOption->filter_depth = 3;

    // 设置滤波器的数量为2：一个用于深度卷积，一个用于逐点卷积
    PLayerOption->filter_number = 2;
    PConvLayer->filters = MakeFilters(PLayerOption->filter_w, PLayerOption->filter_h, PLayerOption->filter_depth, PLayerOption->filter_number);

    // 计算输出的宽度、高度和深度
    PConvLayer->layer.out_w = floor((PConvLayer->layer.in_w + PConvLayer->padding * 2 - PConvLayer->filters->_w) / PConvLayer->stride + 1);
    PConvLayer->layer.out_h = floor((PConvLayer->layer.in_h + PConvLayer->padding * 2 - PConvLayer->filters->_w) / PConvLayer->stride + 1);
    PConvLayer->layer.out_depth = PConvLayer->layer.in_depth; // 输出深度与输入深度一致

    // 初始化输出体积
    PConvLayer->layer.out_v = MakeVolume(PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth);
    PConvLayer->layer.out_v->init(PConvLayer->layer.out_v, PConvLayer->layer.out_w, PConvLayer->layer.out_h, PConvLayer->layer.out_depth, 0);

    // 初始化偏置
    PConvLayer->biases = MakeVolume(1, 1, PConvLayer->layer.out_depth);
    PConvLayer->biases->init(PConvLayer->biases, 1, 1, PConvLayer->layer.out_depth, PConvLayer->bias);

    // 初始化滤波器权重
    for (uint32_t i = 0; i < PConvLayer->layer.out_depth; i++)
    {
        PConvLayer->filters->init(PConvLayer->filters->volumes[i], PConvLayer->filters->_w, PConvLayer->filters->_h, PConvLayer->filters->_depth, 0);
        PConvLayer->filters->volumes[i]->fillGauss(PConvLayer->filters->volumes[i]->weight);
    }

    // 更新输出的宽度、高度和深度参数
    PLayerOption->out_w = PConvLayer->layer.out_w;
    PLayerOption->out_h = PConvLayer->layer.out_h;
    PLayerOption->out_depth = PConvLayer->layer.out_depth;
}

/////////////////////////////////////////////////////////////////////////////
// @brief 深度卷积和逐点卷积的前向计算
// @param PConvLayer - 卷积层指针
void DepthwisePointwiseConvolution(TPConvLayer PConvLayer)
{
    uint16_t padding_x = -PConvLayer->padding;
    uint16_t padding_y = -PConvLayer->padding;
    TPVolume inVolume = PConvLayer->layer.in_v; // 输入体积
    TPVolume outVolume = PConvLayer->layer.out_v; // 输出体积

    // 深度卷积滤波器
    TPVolume depthwiseFilter = PConvLayer->filters->volumes[0];
    // 逐点卷积滤波器
    TPVolume pointwiseFilter = PConvLayer->filters->volumes[1];

    // 深度卷积的运算
    for (uint32_t out_d = 0; out_d < PConvLayer->layer.out_depth; out_d++)
    {
        for (uint32_t out_y = 0; out_y < PConvLayer->layer.out_h; out_y++)
        {
            padding_x = -PConvLayer->padding;
            for (uint32_t out_x = 0; out_x < PConvLayer->layer.out_w; out_x++)
            {
                float32_t sum = 0.0; // 用于存储卷积结果

                // 深度卷积：遍历滤波器的每一行和每一列
                for (uint32_t filter_y = 0; filter_y < PConvLayer->filters->_h; filter_y++)
                {
                    uint32_t in_y = filter_y + padding_y;
                    if (in_y < 0 || in_y >= inVolume->_h)
                        continue;

                    for (uint16_t filter_x = 0; filter_x < PConvLayer->filters->_w; filter_x++)
                    {
                        uint32_t in_x = filter_x + padding_x;
                        if (in_x < 0 || in_x >= inVolume->_w)
                            continue;

                        uint32_t fx = (depthwiseFilter->_w * filter_y + filter_x) * depthwiseFilter->_depth;
                        uint32_t ix = (inVolume->_w * in_y + in_x) * inVolume->_depth;

                        // 按深度进行逐元素乘积计算
                        for (uint32_t filter_d = 0; filter_d < depthwiseFilter->_depth; filter_d++)
                        {
                            sum += depthwiseFilter->weight->buffer[fx + filter_d] * inVolume->weight->buffer[ix + filter_d];
                        }
                    }
                }

                // 将卷积结果存入输出体积
                uint32_t out_idx = (outVolume->_w * out_y + out_x) * outVolume->_depth + out_d;
                outVolume->weight->buffer[out_idx] = sum;

                padding_x += PConvLayer->stride;
            }

            padding_y += PConvLayer->stride;
        }
    }

    // 逐点卷积的运算
    for (uint32_t out_d = 0; out_d < PConvLayer->layer.out_depth; out_d++)
    {
        for (uint32_t out_y = 0; out_y < PConvLayer->layer.out_h; out_y++)
        {
            for (uint32_t out_x = 0; out_x < PConvLayer->layer.out_w; out_x++)
            {
                uint32_t out_idx = (outVolume->_w * out_y + out_x) * outVolume->_depth + out_d;
                float32_t depthwise_out = outVolume->weight->buffer[out_idx]; // 深度卷积的输出
                float32_t pointwise_out = depthwise_out * pointwiseFilter->weight->buffer[out_d]; // 逐点卷积运算
                outVolume->weight->buffer[out_idx] = pointwise_out; // 更新输出体积
            }
        }
    }
}

/// @brief 反向传播，计算关于输入 X、权重 W、偏置 B 的梯度
/// @param PConvLayer - 卷积层指针
/// Y = WX + B，求关于 X、W、B 的偏导数
void ConvolutionLayerBackward(TPConvLayer PConvLayer)
{
    float32_t out_grad = 0.00; // 输出梯度的初始值
    uint16_t padding_x = -PConvLayer->padding; // 水平方向的填充
    uint16_t padding_y = -PConvLayer->padding; // 垂直方向的填充
    uint16_t in_x, in_y; // 输入特征图的坐标
    TPVolume inVolume = PConvLayer->layer.in_v; // 输入特征图
    TPVolume outVolume = PConvLayer->layer.out_v; // 输出特征图

    // 初始化输入特征图的梯度为零
    inVolume->fillZero(inVolume->grads);

    // 遍历输出特征图的每个深度通道
    for (uint32_t out_d = 0; out_d < PConvLayer->layer.out_depth; out_d++)
    {
        TPVolume filter = PConvLayer->filters->volumes[out_d]; // 获取当前深度对应的滤波器
        padding_x = -PConvLayer->padding; // 重置水平方向的填充
        padding_y = -PConvLayer->padding; // 重置垂直方向的填充

        // 遍历输出特征图的每个位置
        for (uint32_t out_y = 0; out_y < PConvLayer->layer.out_h; out_y++)
        {
            padding_x = -PConvLayer->padding;
            for (uint32_t out_x = 0; out_x < PConvLayer->layer.out_w; out_x++)
            {
                // 获取当前输出位置的梯度
                out_grad = outVolume->getGradValue(outVolume, out_x, out_y, out_d);

                // 遍历滤波器的每个位置，计算梯度
                for (uint32_t filter_y = 0; filter_y < PConvLayer->filters->_h; filter_y++)
                {
                    in_y = filter_y + padding_y; // 计算输入特征图中对应的 y 坐标
                    if (in_y < 0 || in_y >= inVolume->_h)
                        continue; // 如果超出边界则跳过

                    for (uint32_t filter_x = 0; filter_x < PConvLayer->filters->_w; filter_x++)
                    {
                        in_x = filter_x + padding_x; // 计算输入特征图中对应的 x 坐标
                        if (in_x < 0 || in_x >= inVolume->_w)
                            continue; // 如果超出边界则跳过

                        // 计算当前滤波器和输入特征图的索引
                        uint32_t fx = (filter->_w * filter_y + filter_x) * filter->_depth;
                        uint32_t ix = (inVolume->_w * in_y + in_x) * inVolume->_depth;

                        // 计算滤波器的梯度和输入特征图的梯度
                        for (uint32_t filter_d = 0; filter_d < PConvLayer->filters->_depth; filter_d++)
                        {
                            // 更新滤波器的梯度
                            filter->grads->buffer[fx + filter_d] += inVolume->weight->buffer[ix + filter_d] * out_grad;
                            // 更新输入特征图的梯度
                            inVolume->grads->buffer[ix + filter_d] += filter->weight->buffer[fx + filter_d] * out_grad;
                        }
                    }
                }

                // 更新偏置的梯度
                PConvLayer->biases->grads->buffer[out_d] += out_grad;

                // 更新水平方向的填充
                padding_x += PConvLayer->stride;
            }

            // 更新垂直方向的填充
            padding_y += PConvLayer->stride;
        }
    }
}

/// @brief 获取卷积层的参数和梯度
/// @param PConvLayer - 卷积层指针
/// @return 包含参数和梯度的数组
TPParameters* ConvolutionLayerGetParamsAndGradients(TPConvLayer PConvLayer)
{
    // 如果输出通道数小于等于 0，返回 NULL
    if (PConvLayer->layer.out_depth <= 0)
        return NULL;

    // 为每个输出通道及偏置分配内存存储参数和梯度
    TPParameters* tPResponses = malloc(sizeof(TPParameters) * (PConvLayer->layer.out_depth + 1));
    if (tPResponses == NULL)
        return NULL;

    // 遍历每个输出通道，获取滤波器的权重和梯度
    for (uint32_t out_d = 0; out_d < PConvLayer->layer.out_depth; out_d++)
    {
        TPParameters PResponse = malloc(sizeof(TParameters));
        if (PResponse != NULL)
        {
            // 获取该通道的滤波器权重和梯度
            PResponse->filterWeight = PConvLayer->filters->volumes[out_d]->weight;
            PResponse->filterGrads = PConvLayer->filters->volumes[out_d]->grads;
            PResponse->l1_decay_rate = PConvLayer->l1_decay_rate; // L1正则化系数
            PResponse->l2_decay_rate = PConvLayer->l2_decay_rate; // L2正则化系数
            PResponse->fillZero = TensorFillZero; // 初始化函数
            PResponse->free = TensorFree; // 释放函数
            tPResponses[out_d] = PResponse;
        }
    }

    // 获取偏置的权重和梯度，并分配内存
    TPParameters PResponse = malloc(sizeof(TParameters));
    if (PResponse != NULL)
    {
        PResponse->filterWeight = PConvLayer->biases->weight;
        PResponse->filterGrads = PConvLayer->biases->grads;
        PResponse->l1_decay_rate = 0; // 偏置没有正则化
        PResponse->l2_decay_rate = 0;
        PResponse->fillZero = TensorFillZero;
        PResponse->free = TensorFree;
        tPResponses[PConvLayer->layer.out_depth] = PResponse; // 将偏置存储在数组的最后一个位置
    }

    return tPResponses; // 返回包含参数和梯度的数组
}

/// @brief 计算卷积层的反向传播损失
/// @param PConvLayer - 卷积层指针
/// @param Y - 目标值
/// @return 损失值
float32_t ConvolutionLayerBackwardLoss(TPConvLayer PConvLayer, int Y)
{
    return 0.00; // 该函数暂未实现，返回 0.00
}

/// @brief 释放卷积层相关的内存
/// @param PConvLayer - 卷积层指针
void ConvolutionLayerFree(TPConvLayer PConvLayer)
{
    // 释放输入、输出、偏置和滤波器的内存
    VolumeFree(PConvLayer->layer.in_v);
    VolumeFree(PConvLayer->layer.out_v);
    VolumeFree(PConvLayer->biases);
    FiltersFree(PConvLayer->filters);

    // 释放滤波器结构体的内存
    if (PConvLayer->filters != NULL)
        free(PConvLayer->filters);

    // 最后释放卷积层本身的内存
    free(PConvLayer);
}

/////////////////////////////////////////////////////////////////////////////
/// @brief //////////////////////////////////////////////////////////////////
/// @param PReluLayer
/// @param PLayerOption
void ReluLayerInit(TPReluLayer PReluLayer, TPLayerOption PLayerOption)
{
    PReluLayer->layer.LayerType = PLayerOption->LayerType;
    // PReluLayer->l1_decay_rate = LayerOption.l1_decay_rate;
    // PReluLayer->l2_decay_rate = LayerOption.l2_decay_rate;
    // PReluLayer->stride = LayerOption.stride;
    // PReluLayer->padding = LayerOption.padding;
    // PReluLayer->bias = LayerOption.bias;

    PReluLayer->layer.in_w = PLayerOption->in_w;
    PReluLayer->layer.in_h = PLayerOption->in_h;
    PReluLayer->layer.in_depth = PLayerOption->in_depth;
    PReluLayer->layer.in_v = NULL;

    PReluLayer->layer.out_w = PReluLayer->layer.in_w;
    PReluLayer->layer.out_h = PReluLayer->layer.in_h;
    PReluLayer->layer.out_depth = PReluLayer->layer.in_depth;

    PReluLayer->layer.out_v = MakeVolume(PReluLayer->layer.out_w, PReluLayer->layer.out_h, PReluLayer->layer.out_depth);
    PReluLayer->layer.out_v->init(PReluLayer->layer.out_v, PReluLayer->layer.out_w, PReluLayer->layer.out_h, PReluLayer->layer.out_depth, 0);

    PLayerOption->out_w = PReluLayer->layer.out_w;
    PLayerOption->out_h = PReluLayer->layer.out_h;
    PLayerOption->out_depth = PReluLayer->layer.out_depth;
}

void reluLayerOutResize(TPReluLayer PReluLayer)
{
    if (PReluLayer->layer.out_w != PReluLayer->layer.in_w || PReluLayer->layer.out_h != PReluLayer->layer.in_h || PReluLayer->layer.out_depth != PReluLayer->layer.in_depth)
    {
        LOGINFOR("ReluLayer resize out_v from %d x %d x %d to %d x %d x %d", PReluLayer->layer.out_w, PReluLayer->layer.out_h, PReluLayer->layer.out_depth, PReluLayer->layer.in_w, PReluLayer->layer.in_h, PReluLayer->layer.in_depth);
        PReluLayer->layer.out_w = PReluLayer->layer.in_w;
        PReluLayer->layer.out_h = PReluLayer->layer.in_h;
        PReluLayer->layer.out_depth = PReluLayer->layer.in_depth;
        if (PReluLayer->layer.out_v != NULL)
        {
            VolumeFree(PReluLayer->layer.out_v);
        }
        PReluLayer->layer.out_v = MakeVolume(PReluLayer->layer.out_w, PReluLayer->layer.out_h, PReluLayer->layer.out_depth);
        PReluLayer->layer.out_v->init(PReluLayer->layer.out_v, PReluLayer->layer.out_w, PReluLayer->layer.out_h, PReluLayer->layer.out_depth, 0);
    }
}

void ReluLayerForward(TPReluLayer PReluLayer)
{
    /*for (uint16_t out_d = 0; out_d < PReluLayer->layer.out_depth; out_d++) {
     for (uint16_t out_y = 0; out_y < PReluLayer->layer.out_h; out_y++) {
     for (uint16_t out_x = 0; out_x < PReluLayer->layer.out_w; out_x++) {
     }
     }
     }*/
    reluLayerOutResize(PReluLayer);
    for (uint32_t out_l = 0; out_l < PReluLayer->layer.out_v->weight->length; out_l++)
    {
        if (PReluLayer->layer.in_v->weight->buffer[out_l] < 0)
            PReluLayer->layer.out_v->weight->buffer[out_l] = 0;
        else
            PReluLayer->layer.out_v->weight->buffer[out_l] = PReluLayer->layer.in_v->weight->buffer[out_l];
    }
}
/// @brief //////////////////////////////////////////////////////////////////////////////////////
/// @param PReluLayer /

void ReluLayerBackward(TPReluLayer PReluLayer)
{
    for (uint32_t out_l = 0; out_l < PReluLayer->layer.in_v->weight->length; out_l++)
    {
        if (PReluLayer->layer.out_v->weight->buffer[out_l] <= 0)
            PReluLayer->layer.in_v->grads->buffer[out_l] = 0;
        else
            PReluLayer->layer.in_v->grads->buffer[out_l] = PReluLayer->layer.out_v->grads->buffer[out_l];
    }
}

float32_t ReluLayerBackwardLoss(TPReluLayer PReluLayer, int Y)
{
    return 0.00;
}

void ReluLayerFree(TPReluLayer PReluLayer)
{
    VolumeFree(PReluLayer->layer.in_v);
    VolumeFree(PReluLayer->layer.out_v);
    free(PReluLayer);
}
////////////////////////////////////////////////////////////////////////////
/// @brief /////////////////////////////////////////////////////////////////
/// @param PPoolLayer
/// @param PLayerOption /
void PoolLayerInit(TPPoolLayer PPoolLayer, TPLayerOption PLayerOption)
{
    PPoolLayer->layer.LayerType = PLayerOption->LayerType;
    // PPoolLayer->l1_decay_rate = PLayerOption->l1_decay_rate;
    // PPoolLayer->l2_decay_rate = PLayerOption->l2_decay_rate;
    PPoolLayer->stride = PLayerOption->stride;
    PPoolLayer->padding = PLayerOption->padding;
    // PPoolLayer->bias = PLayerOption->bias;
    // 池化核不需要分配张量空间
    PPoolLayer->filter = MakeVolume(PLayerOption->filter_w, PLayerOption->filter_h, PLayerOption->filter_depth);

    PPoolLayer->layer.in_w = PLayerOption->in_w;
    PPoolLayer->layer.in_h = PLayerOption->in_h;
    PPoolLayer->layer.in_depth = PLayerOption->in_depth;
    PPoolLayer->layer.in_v = NULL;

    PPoolLayer->layer.out_w = floor((PPoolLayer->layer.in_w + PPoolLayer->padding * 2 - PPoolLayer->filter->_w) / PPoolLayer->stride + 1);
    PPoolLayer->layer.out_h = floor((PPoolLayer->layer.in_h + PPoolLayer->padding * 2 - PPoolLayer->filter->_h) / PPoolLayer->stride + 1);
    PPoolLayer->layer.out_depth = PPoolLayer->layer.in_depth;

    PPoolLayer->layer.out_v = MakeVolume(PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth);
    PPoolLayer->layer.out_v->init(PPoolLayer->layer.out_v, PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, 0.0);

    PPoolLayer->switchxy = MakeVolume(PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth);
    PPoolLayer->switchxy->init(PPoolLayer->switchxy, PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, 0);
    // PPoolLayer->switchy = MakeVolume(PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth);
    // PPoolLayer->switchy->init(PPoolLayer->switchy, PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, 0);
    // PPoolLayer->switchy->free(PPoolLayer->switchy->grads);
    // PPoolLayer->switchx->free(PPoolLayer->switchx->grads);
    // uint16_t out_length = PPoolLayer->layer.out_w * PPoolLayer->layer.out_h * PPoolLayer->layer.out_depth;
    // PPoolLayer->switchx = MakeTensor(out_length);
    // PPoolLayer->switchy = MakeTensor(out_length);

    PLayerOption->out_w = PPoolLayer->layer.out_w;
    PLayerOption->out_h = PPoolLayer->layer.out_h;
    PLayerOption->out_depth = PPoolLayer->layer.out_depth;
}

void poolLayerOutResize(TPPoolLayer PPoolLayer)
{
    uint32_t out_w = floor((PPoolLayer->layer.in_w + PPoolLayer->padding * 2 - PPoolLayer->filter->_w) / PPoolLayer->stride + 1);
    uint32_t out_h = floor((PPoolLayer->layer.in_h + PPoolLayer->padding * 2 - PPoolLayer->filter->_h) / PPoolLayer->stride + 1);
    if (PPoolLayer->layer.out_w != out_w || PPoolLayer->layer.out_h != out_h || PPoolLayer->layer.out_depth != PPoolLayer->layer.in_depth)
    {
        LOGINFOR("PoolLayer resize out_v from %d x %d x %d to %d x %d x %d", PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, out_w, out_h, PPoolLayer->layer.in_depth);
        PPoolLayer->layer.out_w = out_w;
        PPoolLayer->layer.out_h = out_h;
        PPoolLayer->layer.out_depth = PPoolLayer->layer.in_depth;
        if (PPoolLayer->layer.out_v != NULL)
        {
            VolumeFree(PPoolLayer->layer.out_v);
            VolumeFree(PPoolLayer->switchxy);
            // VolumeFree(PPoolLayer->switchy);
        }
        PPoolLayer->layer.out_v = MakeVolume(PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth);
        PPoolLayer->layer.out_v->init(PPoolLayer->layer.out_v, PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, 0);

        PPoolLayer->switchxy = MakeVolume(PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth);
        PPoolLayer->switchxy->init(PPoolLayer->switchxy, PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, 0);
        // PPoolLayer->switchy = MakeVolume(PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth);
        // PPoolLayer->switchy->init(PPoolLayer->switchy, PPoolLayer->layer.out_w, PPoolLayer->layer.out_h, PPoolLayer->layer.out_depth, 0);
        // PPoolLayer->switchy->free(PPoolLayer->switchy->grads);
        // PPoolLayer->switchx->free(PPoolLayer->switchx->grads);
    }
}

void PoolLayerForward(TPPoolLayer PPoolLayer)
{
    float32_t max_value = MINFLOAT_NEGATIVE_NUMBER;
    float32_t value = 0;
    uint16_t x = -PPoolLayer->padding;
    uint16_t y = -PPoolLayer->padding;
    uint16_t ox, oy, inx, iny;
    TPVolume inVolu = PPoolLayer->layer.in_v;
    TPVolume outVolu = PPoolLayer->layer.out_v;
    poolLayerOutResize(PPoolLayer);
    outVolu->fillZero(outVolu->weight);

    for (uint32_t out_d = 0; out_d < PPoolLayer->layer.out_depth; out_d++)
    {
        x = -PPoolLayer->padding;
        y = -PPoolLayer->padding;
        for (uint32_t out_y = 0; out_y < PPoolLayer->layer.out_h; out_y++)
        {
            x = -PPoolLayer->padding;

            for (uint32_t out_x = 0; out_x < PPoolLayer->layer.out_w; out_x++)
            {
                max_value = MINFLOAT_NEGATIVE_NUMBER;
                inx = -1;
                iny = -1;
                for (uint32_t filter_y = 0; filter_y < PPoolLayer->filter->_h; filter_y++)
                {
                    oy = filter_y + y;
                    if (oy < 0 && oy >= inVolu->_h)
                        continue;
                    for (uint32_t filter_x = 0; filter_x < PPoolLayer->filter->_w; filter_x++)
                    {
                        ox = filter_x + x;
                        if (ox >= 0 && ox < inVolu->_w && oy >= 0 && oy < inVolu->_h)
                        {
                            value = inVolu->getValue(inVolu, ox, oy, out_d);
                            if (value > max_value)
                            {
                                max_value = value;
                                inx = ox;
                                iny = oy;
                            }
                        }
                    }
                }
                outVolu->setValue(outVolu, out_x, out_y, out_d, max_value);
                PPoolLayer->switchxy->setValue(PPoolLayer->switchxy, out_x, out_y, out_d, inx);
                PPoolLayer->switchxy->setGradValue(PPoolLayer->switchxy, out_x, out_y, out_d, iny);
                x = x + PPoolLayer->stride;
            }
            y = y + PPoolLayer->stride;
        }
    }
}
/// @brief ////////////////////////////////////////////////////////////////////////////////////
/// @param PPoolLayer /
///
void PoolLayerBackward(TPPoolLayer PPoolLayer)
{
    float32_t grad_value = 0.00;
    TPVolume inVolu = PPoolLayer->layer.in_v;
    TPVolume outVolu = PPoolLayer->layer.out_v;
    inVolu->fillZero(inVolu->grads);
    uint32_t x, y;
    for (uint32_t out_d = 0; out_d < PPoolLayer->layer.out_depth; out_d++)
    {
        x = -PPoolLayer->padding;
        y = -PPoolLayer->padding;
        for (uint32_t out_y = 0; out_y < PPoolLayer->layer.out_h; out_y++)
        {
            x = -PPoolLayer->padding;
            for (uint32_t out_x = 0; out_x < PPoolLayer->layer.out_w; out_x++)
            {
                grad_value = outVolu->getGradValue(outVolu, out_x, out_y, out_d);
                uint32_t ox = PPoolLayer->switchxy->getValue(PPoolLayer->switchxy, out_x, out_y, out_d);
                uint32_t oy = PPoolLayer->switchxy->getGradValue(PPoolLayer->switchxy, out_x, out_y, out_d);
                inVolu->addGradValue(inVolu, ox, oy, out_d, grad_value);
                x = x + PPoolLayer->stride;
            }
            y = y + PPoolLayer->stride;
        }
    }
}

float32_t PoolLayerBackwardLoss(TPPoolLayer PPoolLayer, int Y)
{
    return 0.00;
}

void PoolLayerFree(TPPoolLayer PPoolLayer)
{
    VolumeFree(PPoolLayer->layer.in_v);
    VolumeFree(PPoolLayer->layer.out_v);
    VolumeFree(PPoolLayer->switchxy);
    // VolumeFree(PPoolLayer->switchy);
    free(PPoolLayer);
}
////////////////////////////////////////////////////////////////////////////////
// FullyConnLayer
/// @brief ////////////////////////////////////////////////////////////////////
/// @param PFullyConnLayer
/// @param PLayerOption
void FullyConnLayerInit(TPFullyConnLayer PFullyConnLayer, TPLayerOption PLayerOption)
{
    PFullyConnLayer->layer.LayerType = PLayerOption->LayerType;
    PFullyConnLayer->l1_decay_rate = PLayerOption->l1_decay_rate;
    PFullyConnLayer->l2_decay_rate = PLayerOption->l2_decay_rate;
    // PFullyConnLayer->stride = LayerOption.stride;
    // PFullyConnLayer->padding = LayerOption.padding;
    PFullyConnLayer->bias = PLayerOption->bias;
    // PFullyConnLayer->filter._w = LayerOption.filter_w;
    // PFullyConnLayer->filter._h = LayerOption.filter_h;
    // PFullyConnLayer->filter._depth = LayerOption.filter_depth;
    PLayerOption->filter_w = 1;
    PLayerOption->filter_h = 1;

    PFullyConnLayer->layer.in_w = PLayerOption->in_w;
    PFullyConnLayer->layer.in_h = PLayerOption->in_h;
    PFullyConnLayer->layer.in_depth = PLayerOption->in_depth;
    PFullyConnLayer->layer.in_v = NULL;

    uint32_t inputPoints = PFullyConnLayer->layer.in_w * PFullyConnLayer->layer.in_h * PFullyConnLayer->layer.in_depth;

    PFullyConnLayer->filters = MakeFilters(PLayerOption->filter_w, PLayerOption->filter_h, inputPoints, PLayerOption->filter_number);

    for (uint32_t i = 0; i < PFullyConnLayer->filters->filterNumber; i++)
    {
        PFullyConnLayer->filters->init(PFullyConnLayer->filters->volumes[i], PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, inputPoints, 0);
        PFullyConnLayer->filters->volumes[i]->fillGauss(PFullyConnLayer->filters->volumes[i]->weight);
    }

    PFullyConnLayer->layer.out_w = 1;
    PFullyConnLayer->layer.out_h = 1;
    PFullyConnLayer->layer.out_depth = PFullyConnLayer->filters->filterNumber;
    PFullyConnLayer->layer.out_v = MakeVolume(PFullyConnLayer->layer.out_w, PFullyConnLayer->layer.out_h, PFullyConnLayer->layer.out_depth);
    PFullyConnLayer->layer.out_v->init(PFullyConnLayer->layer.out_v, PFullyConnLayer->layer.out_w, PFullyConnLayer->layer.out_h,
                                       PFullyConnLayer->layer.out_depth, 0);

    PFullyConnLayer->biases = MakeVolume(1, 1, PFullyConnLayer->layer.out_depth);
    PFullyConnLayer->biases->init(PFullyConnLayer->biases, 1, 1, PFullyConnLayer->layer.out_depth, PFullyConnLayer->bias);

    PLayerOption->out_w = PFullyConnLayer->layer.out_w;
    PLayerOption->out_h = PFullyConnLayer->layer.out_h;
    PLayerOption->out_depth = PFullyConnLayer->layer.out_depth;
}

void fullConnLayerOutResize(TPFullyConnLayer PFullyConnLayer)
{
    uint32_t inputLength = PFullyConnLayer->layer.in_w * PFullyConnLayer->layer.in_h * PFullyConnLayer->layer.in_depth;
    if (PFullyConnLayer->filters->_depth != inputLength)
    {
        LOGINFOR("FullyConnLayer resize filters from %d x %d x %d to %d x %d x %d", PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, PFullyConnLayer->filters->_depth, PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, inputLength);
        FiltersFree(PFullyConnLayer->filters);
        bool ret = FiltersResize(PFullyConnLayer->filters, PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, inputLength, PFullyConnLayer->filters->filterNumber);
        if (ret)
        {
            for (uint16_t i = 0; i < PFullyConnLayer->filters->filterNumber; i++)
            {
                PFullyConnLayer->filters->init(PFullyConnLayer->filters->volumes[i], PFullyConnLayer->filters->_w, PFullyConnLayer->filters->_h, inputLength, 0);
                PFullyConnLayer->filters->volumes[i]->fillGauss(PFullyConnLayer->filters->volumes[i]->weight);
            }
        }
    }
}
void FullyConnLayerForward(TPFullyConnLayer PFullyConnLayer)
{
    float32_t sum = 0.00;
    TPVolume inVolu = PFullyConnLayer->layer.in_v;
    TPVolume outVolu = PFullyConnLayer->layer.out_v;
    outVolu->fillZero(outVolu->weight);
    fullConnLayerOutResize(PFullyConnLayer);

    for (uint32_t out_d = 0; out_d < PFullyConnLayer->layer.out_depth; out_d++)
    {
        TPVolume filter = PFullyConnLayer->filters->volumes[out_d];
        sum = 0.00;

        uint32_t inputPoints = inVolu->_w * inVolu->_h * inVolu->_depth;

        for (uint32_t ip = 0; ip < inputPoints; ip++)
        {
            if (filter->weight->length == 0)
                sum = sum + inVolu->weight->buffer[ip];
            else
                sum = sum + inVolu->weight->buffer[ip] * filter->weight->buffer[ip];
        }

        sum = sum + PFullyConnLayer->biases->weight->buffer[out_d];
        outVolu->weight->buffer[out_d] = sum;
    }
}

/// @brief /////////////////////////////////////////////////////////////////////////////////////////////////
/// @param PFullyConnLayer
/// y = wx+b 求关于in w b的偏导数
void FullyConnLayerBackward(TPFullyConnLayer PFullyConnLayer)
{
    float32_t grad_value = 0.00;
    TPVolume inVolu = PFullyConnLayer->layer.in_v;
    TPVolume outVolu = PFullyConnLayer->layer.out_v;
    inVolu->fillZero(inVolu->grads);

    for (uint32_t out_d = 0; out_d < PFullyConnLayer->layer.out_depth; out_d++)
    {
        TPVolume filter = PFullyConnLayer->filters->volumes[out_d];
        grad_value = outVolu->grads->buffer[out_d];

        uint32_t inputPoints = inVolu->_w * inVolu->_h * inVolu->_depth;

        for (uint32_t out_l = 0; out_l < inputPoints; out_l++)
        {
            inVolu->grads->buffer[out_l] = inVolu->grads->buffer[out_l] + filter->weight->buffer[out_l] * grad_value;
            filter->grads->buffer[out_l] = filter->grads->buffer[out_l] + inVolu->weight->buffer[out_l] * grad_value;
        }
        PFullyConnLayer->biases->grads->buffer[out_d] = PFullyConnLayer->biases->grads->buffer[out_d] * grad_value;
    }
}

float32_t FullyConnLayerBackwardLoss(TPFullyConnLayer PFullyConnLayer, int Y)
{
    return 0.00;
}

TPParameters* FullyConnLayerGetParamsAndGrads(TPFullyConnLayer PFullyConnLayer)
{
    if (PFullyConnLayer->layer.out_depth <= 0)
        return NULL;
    TPParameters* tPResponses = malloc(sizeof(TPParameters) * (PFullyConnLayer->layer.out_depth + 1));
    if (tPResponses == NULL)
        return NULL;
    for (uint32_t out_d = 0; out_d < PFullyConnLayer->layer.out_depth; out_d++)
    {
        TPParameters PResponse = malloc(sizeof(TParameters));
        if (PResponse != NULL)
        {
            PResponse->filterWeight = PFullyConnLayer->filters->volumes[out_d]->weight;
            PResponse->filterGrads = PFullyConnLayer->filters->volumes[out_d]->grads;
            PResponse->l1_decay_rate = PFullyConnLayer->l1_decay_rate;
            PResponse->l2_decay_rate = PFullyConnLayer->l2_decay_rate;
            PResponse->fillZero = TensorFillZero;
            PResponse->free = TensorFree;
            tPResponses[out_d] = PResponse;
        }
    }
    TPParameters PResponse = malloc(sizeof(TParameters));
    if (PResponse != NULL)
    {
        PResponse->filterWeight = PFullyConnLayer->biases->weight;
        PResponse->filterGrads = PFullyConnLayer->biases->grads;
        PResponse->l1_decay_rate = 0;
        PResponse->l2_decay_rate = 0;
        PResponse->fillZero = TensorFillZero;
        PResponse->free = TensorFree;
        tPResponses[PFullyConnLayer->layer.out_depth] = PResponse;
    }
    return tPResponses;
}

void FullyConnLayerFree(TPFullyConnLayer PFullyConnLayer)
{
    VolumeFree(PFullyConnLayer->layer.in_v);
    VolumeFree(PFullyConnLayer->layer.out_v);
    VolumeFree(PFullyConnLayer->biases);
    FiltersFree(PFullyConnLayer->filters);
    // for (uint16_t i = 0; i < PFullyConnLayer->filters->_depth; i++)
    //{
    //	PFullyConnLayer->filters->free(PFullyConnLayer->filters->volumes[i]);
    // }
    if (PFullyConnLayer->filters != NULL)
        free(PFullyConnLayer->filters);
    free(PFullyConnLayer);
}

////////////////////////////////////////////////////////////////////////////////
// Softmax
void SoftmaxLayerInit(TPSoftmaxLayer PSoftmaxLayer, TPLayerOption PLayerOption)
{
    PSoftmaxLayer->layer.LayerType = PLayerOption->LayerType;
    PSoftmaxLayer->layer.in_w = PLayerOption->in_w;
    PSoftmaxLayer->layer.in_h = PLayerOption->in_h;
    PSoftmaxLayer->layer.in_depth = PLayerOption->in_depth;
    PSoftmaxLayer->layer.in_v = NULL;

    PSoftmaxLayer->layer.out_w = 1;
    PSoftmaxLayer->layer.out_h = 1;
    PSoftmaxLayer->layer.out_depth = PLayerOption->in_w * PLayerOption->in_h * PLayerOption->in_depth;

    PSoftmaxLayer->layer.out_v = MakeVolume(PSoftmaxLayer->layer.out_w, PSoftmaxLayer->layer.out_h, PSoftmaxLayer->layer.out_depth);
    PSoftmaxLayer->layer.out_v->init(PSoftmaxLayer->layer.out_v, PSoftmaxLayer->layer.out_w, PSoftmaxLayer->layer.out_h, PSoftmaxLayer->layer.out_depth, 0);
    PSoftmaxLayer->exp = MakeTensor(PSoftmaxLayer->layer.out_depth);

    PLayerOption->out_w = PSoftmaxLayer->layer.out_w;
    PLayerOption->out_h = PSoftmaxLayer->layer.out_h;
    PLayerOption->out_depth = PSoftmaxLayer->layer.out_depth;
}
/// @brief ////////////////////////////////////////////////////////////////////////
/// @param PSoftmaxLayer
void softmaxLayOutResize(TPSoftmaxLayer PSoftmaxLayer)
{
    uint16_t inputLength = PSoftmaxLayer->layer.in_depth * PSoftmaxLayer->layer.in_w * PSoftmaxLayer->layer.in_h;
    if (PSoftmaxLayer->layer.out_depth != inputLength)
    {
        LOGINFOR("Softmax resize out_v from %d x %d x %d to %d x %d x %d", PSoftmaxLayer->layer.out_w, PSoftmaxLayer->layer.out_h, PSoftmaxLayer->layer.out_depth, 1, 1, inputLength);
        PSoftmaxLayer->layer.out_w = 1;
        PSoftmaxLayer->layer.out_h = 1;
        PSoftmaxLayer->layer.out_depth = inputLength;

        if (PSoftmaxLayer->layer.out_v != NULL)
        {
            VolumeFree(PSoftmaxLayer->layer.out_v);
            TensorFree(PSoftmaxLayer->exp);
        }
        PSoftmaxLayer->layer.out_v = MakeVolume(PSoftmaxLayer->layer.out_w, PSoftmaxLayer->layer.out_h, PSoftmaxLayer->layer.out_depth);
        PSoftmaxLayer->layer.out_v->init(PSoftmaxLayer->layer.out_v, PSoftmaxLayer->layer.out_w, PSoftmaxLayer->layer.out_h, PSoftmaxLayer->layer.out_depth, 0);
        PSoftmaxLayer->exp = MakeTensor(PSoftmaxLayer->layer.out_depth);
    }
}
/// @brief ////////////////////////////////////////////////////////////////////
/// @param PSoftmaxLayer
/// PSoftmaxLayer->exp Probability Distribution
/// PSoftmaxLayer->layer.out_v->weight Probability Distribution
// 归一化后的真数当作概率分布
void SoftmaxLayerForward(TPSoftmaxLayer PSoftmaxLayer)
{
    float32_t max_value = MINFLOAT_NEGATIVE_NUMBER;
    float32_t sum = 0.0;
    float32_t expv = 0.0;
    float32_t temp = 0.0;
    TPVolume inVolu = PSoftmaxLayer->layer.in_v;
    TPVolume outVolu = PSoftmaxLayer->layer.out_v;

    softmaxLayOutResize(PSoftmaxLayer);
    outVolu->fillZero(outVolu->weight);

    for (uint32_t out_d = 0; out_d < PSoftmaxLayer->layer.out_depth; out_d++)
    {
        if (inVolu->weight->buffer[out_d] > max_value)
        {
            max_value = inVolu->weight->buffer[out_d];
        }
    }
    for (uint32_t out_d = 0; out_d < PSoftmaxLayer->layer.out_depth; out_d++)
    {
        temp = inVolu->weight->buffer[out_d] - max_value;
        expv = exp(temp);
        PSoftmaxLayer->exp->buffer[out_d] = expv;
        sum = sum + expv;
    }
    for (uint32_t out_d = 0; out_d < PSoftmaxLayer->layer.out_depth; out_d++)
    {
        PSoftmaxLayer->exp->buffer[out_d] = PSoftmaxLayer->exp->buffer[out_d] / sum;
        PSoftmaxLayer->layer.out_v->weight->buffer[out_d] = PSoftmaxLayer->exp->buffer[out_d];
    }
}
/// @brief ///////////////////////////////////////////////////////////////////////////////
/// @param PSoftmaxLayer
/// dw为代价函数的值，训练输出与真实值之间差
void SoftmaxLayerBackward(TPSoftmaxLayer PSoftmaxLayer)
{
    TPVolume inVolu = PSoftmaxLayer->layer.in_v;
    TPVolume outVolu = PSoftmaxLayer->layer.out_v;
    inVolu->fillZero(outVolu->grads);
    float32_t dw; // 计算 delta weight
    for (uint32_t out_d = 0; out_d < PSoftmaxLayer->layer.out_depth; out_d++)
    {
        if (out_d == PSoftmaxLayer->expected_value)
            dw = -(1 - PSoftmaxLayer->exp->buffer[out_d]);
        else
            dw = PSoftmaxLayer->exp->buffer[out_d];

        inVolu->grads->buffer[out_d] = dw;
    }

    //float32_t exp = PSoftmaxLayer->exp->buffer[PSoftmaxLayer->expected_value];
    //float32_t loss = 0;
    //if (exp > 0)
    //	loss = -log10(exp);

}
/// @brief ////////////////////////////////////////////////////////////////////////////
/// @param PSoftmaxLayer
/// @return
// 交叉熵Cross-Entropy损失函数，衡量模型输出的概率分布与真实标签的差异。
// 交叉熵的计算公式如下：
// CE = -Σylog(ŷ)
// 其中，y是真实标签的概率分布，ŷ是模型输出的概率分布。
// 训练使得损失函数的值无限逼近0，对应的幂exp无限接近1
float32_t SoftmaxLayerBackwardLoss(TPSoftmaxLayer PSoftmaxLayer)
{
    float32_t exp = PSoftmaxLayer->exp->buffer[PSoftmaxLayer->expected_value];
    if (exp > 0)
        return -log10(exp);
    else
        return 0;
}

void SoftmaxLayerFree(TPSoftmaxLayer PSoftmaxLayer)
{
    VolumeFree(PSoftmaxLayer->layer.in_v);
    VolumeFree(PSoftmaxLayer->layer.out_v);
    TensorFree(PSoftmaxLayer->exp);
    free(PSoftmaxLayer);
}

////////////////////////////////////////////////////////////////////////////////////
/// @brief /////////////////////////////////////////////////////////////////////////
/// @param PNeuralNet
/// @param PLayerOption
void NeuralNetInit(TPNeuralNet PNeuralNet, TPLayerOption PLayerOption)
{
    if (PNeuralNet == NULL)
        return;
    void* pLayer = NULL;
    switch (PLayerOption->LayerType)
    {
    case Layer_Type_Input:
    {
        PNeuralNet->depth = 1;
        // free(PNeuralNet->layers);
        PNeuralNet->layers = malloc(sizeof(TPLayer) * PNeuralNet->depth);
        if (PNeuralNet->layers != NULL)
        {
            TPInputLayer InputLayer = malloc(sizeof(TInputLayer));
            InputLayer->init = InputLayerInit;
            InputLayer->free = InputLayerFree;
            InputLayer->forward = InputLayerForward;
            InputLayer->backward = InputLayerBackward;
            InputLayer->computeLoss = NULL; // InputLayerBackwardLoss;
            // InputLayer->backwardOutput = NULL; // InputLayerBackwardOutput;
            InputLayerInit(InputLayer, PLayerOption);
            PNeuralNet->layers[PNeuralNet->depth - 1] = (TPLayer)InputLayer;
        }
        break;
    }
    case Layer_Type_Convolution:
    {
        PNeuralNet->depth++;
        pLayer = realloc(PNeuralNet->layers, sizeof(TPLayer) * PNeuralNet->depth);
        if (pLayer == NULL)
            break;
        PNeuralNet->layers = pLayer;
        if (PNeuralNet->layers != NULL)
        {
            TPConvLayer ConvLayer = malloc(sizeof(TConvLayer));
            ConvLayer->init = ConvolutionLayerInit;
            ConvLayer->free = ConvolutionLayerFree;
            ConvLayer->forward = ConvolutionLayerForward;
            ConvLayer->backward = ConvolutionLayerBackward;
            ConvLayer->computeLoss = ConvolutionLayerBackwardLoss;
            // ConvLayer->backwardOutput = ConvolutionLayerBackwardOutput;
            ConvLayer->getWeightsAndGrads = ConvolutionLayerGetParamsAndGradients;
            ConvolutionLayerInit(ConvLayer, PLayerOption);
            PNeuralNet->layers[PNeuralNet->depth - 1] = (TPLayer)ConvLayer;
        }
        break;
    }
    case Layer_Type_Pool:
    {
        PNeuralNet->depth++;
        pLayer = realloc(PNeuralNet->layers, sizeof(TPLayer) * PNeuralNet->depth);
        if (pLayer == NULL)
            break;
        PNeuralNet->layers = pLayer;
        if (PNeuralNet->layers != NULL)
        {
            TPPoolLayer PoolLayer = malloc(sizeof(TPoolLayer));
            PoolLayer->init = PoolLayerInit;
            PoolLayer->free = PoolLayerFree;
            PoolLayer->forward = PoolLayerForward;
            PoolLayer->backward = PoolLayerBackward;
            PoolLayer->computeLoss = PoolLayerBackwardLoss;
            // PoolLayer->backwardOutput = PoolLayerBackwardOutput;
            PoolLayerInit(PoolLayer, PLayerOption);
            PNeuralNet->layers[PNeuralNet->depth - 1] = (TPLayer)PoolLayer;
        }
        break;
    }
    case Layer_Type_ReLu:
    {
        PNeuralNet->depth++;
        pLayer = realloc(PNeuralNet->layers, sizeof(TPLayer) * PNeuralNet->depth);
        if (pLayer == NULL)
            break;
        PNeuralNet->layers = pLayer;
        if (PNeuralNet->layers != NULL)
        {
            TPReluLayer ReluLayer = malloc(sizeof(TReluLayer));
            ReluLayer->init = ReluLayerInit;
            ReluLayer->free = ReluLayerFree;
            ReluLayer->forward = ReluLayerForward;
            ReluLayer->backward = ReluLayerBackward;
            ReluLayer->computeLoss = ReluLayerBackwardLoss;
            // ReluLayer->backwardOutput = ReluLayerBackwardOutput;
            ReluLayerInit(ReluLayer, PLayerOption);
            PNeuralNet->layers[PNeuralNet->depth - 1] = (TPLayer)ReluLayer;
        }
        break;
    }
    case Layer_Type_FullyConnection:
    {
        PNeuralNet->depth++;
        pLayer = realloc(PNeuralNet->layers, sizeof(TPLayer) * PNeuralNet->depth);
        if (pLayer == NULL)
            break;
        PNeuralNet->layers = pLayer;
        if (PNeuralNet->layers != NULL)
        {
            TPFullyConnLayer FullyConnLayer = malloc(sizeof(TFullyConnLayer));
            FullyConnLayer->init = FullyConnLayerInit;
            FullyConnLayer->free = FullyConnLayerFree;
            FullyConnLayer->forward = FullyConnLayerForward;
            FullyConnLayer->backward = FullyConnLayerBackward;
            FullyConnLayer->computeLoss = FullyConnLayerBackwardLoss;
            // FullyConnLayer->backwardOutput = FullyConnLayerBackwardOutput;
            FullyConnLayer->getWeightsAndGrads = FullyConnLayerGetParamsAndGrads;
            FullyConnLayerInit(FullyConnLayer, PLayerOption);
            PNeuralNet->layers[PNeuralNet->depth - 1] = (TPLayer)FullyConnLayer;
        }
        break;
    }
    case Layer_Type_SoftMax:
    {
        PNeuralNet->depth++;
        pLayer = realloc(PNeuralNet->layers, sizeof(TPLayer) * PNeuralNet->depth);
        if (pLayer == NULL)
            break;
        PNeuralNet->layers = pLayer;
        if (PNeuralNet->layers != NULL)
        {
            TPSoftmaxLayer SoftmaxLayer = malloc(sizeof(TSoftmaxLayer));
            SoftmaxLayer->init = SoftmaxLayerInit;
            SoftmaxLayer->free = SoftmaxLayerFree;
            SoftmaxLayer->forward = SoftmaxLayerForward;
            SoftmaxLayer->backward = SoftmaxLayerBackward;
            SoftmaxLayer->computeLoss = SoftmaxLayerBackwardLoss;
            // SoftmaxLayer.backwardOutput = SoftmaxLayerBackwardOutput;
            SoftmaxLayerInit(SoftmaxLayer, PLayerOption);
            PNeuralNet->layers[PNeuralNet->depth - 1] = (TPLayer)SoftmaxLayer;
        }
        break;
    }
    default:
    {
        break;
    }
    }
}

void NeuralNetFree(TPNeuralNet PNeuralNet)
{
    if (PNeuralNet == NULL)
        return;
    for (uint16_t layerIndex = 0; layerIndex < PNeuralNet->depth; layerIndex++)
    {
        switch (PNeuralNet->layers[layerIndex]->LayerType)
        {
        case Layer_Type_Input:
        {
            InputLayerFree((TPInputLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_Convolution:
        {
            ConvolutionLayerFree((TPConvLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_Pool:
        {
            PoolLayerFree((TPPoolLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_ReLu:
        {
            ReluLayerFree((TPReluLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_FullyConnection:
        {
            FullyConnLayerFree((TPFullyConnLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_SoftMax:
        {
            SoftmaxLayerFree((TPSoftmaxLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        default:
        {
            break;
        }
        }
    }
    free(PNeuralNet);
    PNeuralNet = NULL;
}

void NeuralNetForward(TPNeuralNet PNeuralNet, TPVolume PVolume)
{
    if (PVolume == NULL)
        return;
    TPVolume out_v = NULL;
    TPInputLayer PInputLayer = (TPInputLayer)PNeuralNet->layers[0];
    if (PInputLayer->layer.LayerType != Layer_Type_Input)
    {
        return;
    }

    PInputLayer->forward(PInputLayer, PVolume);

    for (uint16_t layerIndex = 1; layerIndex < PNeuralNet->depth; layerIndex++)
    {
        // PNeuralNet->layers[layerIndex]->in_v->_w = PNeuralNet->layers[layerIndex - 1]->out_w;
        // PNeuralNet->layers[layerIndex]->in_v->_h = PNeuralNet->layers[layerIndex - 1]->out_h;
        // PNeuralNet->layers[layerIndex]->in_v->_depth = PNeuralNet->layers[layerIndex - 1]->out_depth;
        out_v = PNeuralNet->layers[layerIndex - 1]->out_v;
        if (out_v == NULL)
        {
            LOGERROR("Input volume is null, layerIndex=%d layereType=%d\n", layerIndex, PNeuralNet->layers[layerIndex - 1]->LayerType);
            break;
        }

        PNeuralNet->layers[layerIndex]->in_v = out_v;

        switch (PNeuralNet->layers[layerIndex]->LayerType)
        {
        case Layer_Type_Input:
        {
            //((TPInputLayer) PNeuralNet->Layers[layerIndex])->forward((TPInputLayer) PNeuralNet->Layers[layerIndex]);
            break;
        }
        case Layer_Type_Convolution:
        {
            ((TPConvLayer)PNeuralNet->layers[layerIndex])->forward((TPConvLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_Pool:
        {
            ((TPPoolLayer)PNeuralNet->layers[layerIndex])->forward((TPPoolLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_ReLu:
        {
            ((TPReluLayer)PNeuralNet->layers[layerIndex])->forward((TPReluLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_FullyConnection:
        {
            ((TPFullyConnLayer)PNeuralNet->layers[layerIndex])->forward((TPFullyConnLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_SoftMax:
        {
            ((TPSoftmaxLayer)PNeuralNet->layers[layerIndex])->forward((TPSoftmaxLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        default:
            break;
        }
    }
}

void NeuralNetBackward(TPNeuralNet PNeuralNet)
{

    for (uint16_t layerIndex = PNeuralNet->depth - 1; layerIndex >= 0; layerIndex--)
    {
        switch (PNeuralNet->layers[layerIndex]->LayerType)
        {
        case Layer_Type_Input:
        {
            ((TPInputLayer)PNeuralNet->layers[layerIndex])->backward((TPInputLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_Convolution:
        {
            ((TPConvLayer)PNeuralNet->layers[layerIndex])->backward((TPConvLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_Pool:
        {
            ((TPPoolLayer)PNeuralNet->layers[layerIndex])->backward((TPPoolLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_ReLu:
        {
            ((TPReluLayer)PNeuralNet->layers[layerIndex])->backward((TPReluLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_FullyConnection:
        {
            ((TPFullyConnLayer)PNeuralNet->layers[layerIndex])->backward((TPFullyConnLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        case Layer_Type_SoftMax:
        {
            ((TPSoftmaxLayer)PNeuralNet->layers[layerIndex])->backward((TPSoftmaxLayer)PNeuralNet->layers[layerIndex]);
            break;
        }
        default:
            break;
        }
    }
}

void NeuralNetGetWeightsAndGrads(TPNeuralNet PNeuralNet)
{
    bool weight_flow = false;
    bool grads_flow = false;
    void* temp = NULL;
    if (PNeuralNet->trainning.pResponseResults != NULL)
    {
        free(PNeuralNet->trainning.pResponseResults);
        PNeuralNet->trainning.responseCount = 0;
        PNeuralNet->trainning.pResponseResults = NULL;
    }
    if (PNeuralNet->trainning.pResponseResults == NULL)
        temp = malloc(sizeof(TPParameters));

    for (uint16_t layerIndex = 1; layerIndex < PNeuralNet->depth; layerIndex++)
    {
        switch (PNeuralNet->layers[layerIndex]->LayerType)
        {
        case Layer_Type_Input:
        {
            // pResponseResult = ((TPInputLayer) PNeuralNet->layers[layerIndex])->getWeightsAndGrads();
            break;
        }

        case Layer_Type_Convolution:
        {
            TPParameters* pResponseResult = ((TPConvLayer)PNeuralNet->layers[layerIndex])->getWeightsAndGrads(((TPConvLayer)PNeuralNet->layers[layerIndex]));
            for (uint16_t i = 0; i <= ((TPConvLayer)PNeuralNet->layers[layerIndex])->layer.out_depth; i++)
            {
                temp = realloc(PNeuralNet->trainning.pResponseResults, sizeof(TPParameters) * (PNeuralNet->trainning.responseCount + 1));
                if (temp != NULL)
                {
#if 0
                    for (uint16_t w_l = 0; w_l < pResponseResult[i]->filterWeight->length; w_l++)
                    {
                        if (IsFloatOverflow(pResponseResult[i]->filterWeight->buffer[w_l]))
                        {
                            PNeuralNet->trainning.overflow = true;
                            weight_flow = true;
                        }
                    }
                    for (uint16_t w_l = 0; w_l < pResponseResult[i]->filterGrads->length; w_l++)
                    {
                        if (IsFloatOverflow(pResponseResult[i]->filterGrads->buffer[w_l]))
                        {
                            PNeuralNet->trainning.overflow = true;
                            grads_flow = true;
                        }
                    }
#endif
                    PNeuralNet->trainning.pResponseResults = temp;
                    PNeuralNet->trainning.pResponseResults[PNeuralNet->trainning.responseCount] = pResponseResult[i];
                    PNeuralNet->trainning.responseCount++;
                }
            }
            if (PNeuralNet->trainning.underflow)
                LOGERROR("filterGrads underflow %s %d", NeuralNetGetLayerName(Layer_Type_Convolution), layerIndex);
            if (PNeuralNet->trainning.overflow)
            {
                if (grads_flow)
                    LOGERROR("filterGrads overflow %s %d", NeuralNetGetLayerName(Layer_Type_Convolution), layerIndex);
                if (weight_flow)
                    LOGERROR("filterWeight overflow %s %d", NeuralNetGetLayerName(Layer_Type_Convolution), layerIndex);
            }
            free(pResponseResult);
            break;
        }

        case Layer_Type_Pool:
        {
            // pResponseResult = ((TPPoolLayer) PNeuralNet->layers[layerIndex])->getWeightsAndGrads();
            break;
        }

        case Layer_Type_ReLu:
        {
            // pResponseResult = ((TPReluLayer) PNeuralNet->layers[layerIndex])->getWeightsAndGrads();
            break;
        }

        case Layer_Type_FullyConnection:
        {
            TPParameters* pResponseResult = ((TPFullyConnLayer)PNeuralNet->layers[layerIndex])->getWeightsAndGrads(((TPFullyConnLayer)PNeuralNet->layers[layerIndex]));
            for (uint16_t i = 0; i <= ((TPConvLayer)PNeuralNet->layers[layerIndex])->layer.out_depth; i++)
            {
                temp = realloc(PNeuralNet->trainning.pResponseResults, sizeof(TPParameters) * (PNeuralNet->trainning.responseCount + 1));
                if (temp != NULL)
                {
#if 0
                    for (uint16_t w_l = 0; w_l < pResponseResult[i]->filterWeight->length; w_l++)
                    {
                        if (IsFloatOverflow(pResponseResult[i]->filterWeight->buffer[w_l]))
                        {
                            PNeuralNet->trainning.overflow = true;
                            weight_flow = true;
                        }
                    }
                    for (uint16_t w_l = 0; w_l < pResponseResult[i]->filterWeight->length; w_l++)
                    {
                        if (IsFloatOverflow(pResponseResult[i]->filterGrads->buffer[w_l]))
                        {
                            PNeuralNet->trainning.overflow = true;
                            grads_flow = true;
                        }
                    }
#endif
                    PNeuralNet->trainning.pResponseResults = temp;
                    PNeuralNet->trainning.pResponseResults[PNeuralNet->trainning.responseCount] = pResponseResult[i];
                    PNeuralNet->trainning.responseCount++;
                }
            }

            if (PNeuralNet->trainning.underflow)
                LOGERROR("filterGrads underflow %s %d", NeuralNetGetLayerName(Layer_Type_FullyConnection), layerIndex);
            if (PNeuralNet->trainning.overflow)
            {
                if (grads_flow)
                    LOGERROR("filterGrads overflow %s %d", NeuralNetGetLayerName(Layer_Type_FullyConnection), layerIndex);
                if (weight_flow)
                    LOGERROR("filterWeight overflow %s %d", NeuralNetGetLayerName(Layer_Type_FullyConnection), layerIndex);
            }
            free(pResponseResult);
            break;
        }

        case Layer_Type_SoftMax:
        {
            // pResponseResult = ((TPSoftmaxLayer) PNeuralNet->Layers[layerIndex])->getWeightsAndGrads();
            break;
        }
        default:
            break;
        }
    }
}

void NeuralNetComputeCostLoss(TPNeuralNet PNeuralNet, float32_t* CostLoss)
{
    TPSoftmaxLayer PSoftmaxLayer = ((TPSoftmaxLayer)PNeuralNet->layers[PNeuralNet->depth - 1]);
    PSoftmaxLayer->expected_value = PNeuralNet->trainning.labelIndex;
    *CostLoss = PSoftmaxLayer->computeLoss(PSoftmaxLayer);
}

void NeuralNetGetMaxPrediction(TPNeuralNet PNeuralNet, TPPrediction PPrediction)
{
    float32_t maxv = -1;
    uint16_t maxi = -1;
    // TPrediction *pPrediction = malloc(sizeof(TPrediction));
    TPSoftmaxLayer PSoftmaxLayer = ((TPSoftmaxLayer)PNeuralNet->layers[PNeuralNet->depth - 1]);
    for (uint16_t i = 0; i < PSoftmaxLayer->layer.out_v->weight->length; i++)
    {
        if (PSoftmaxLayer->layer.out_v->weight->buffer[i] > maxv)
        {
            maxv = PSoftmaxLayer->layer.out_v->weight->buffer[i];
            maxi = i;
        }
    }
    PPrediction->labelIndex = maxi;
    PPrediction->likeliHood = maxv;
}

void NeuralNetUpdatePrediction(TPNeuralNet PNeuralNet)
{
    const uint16_t defaultPredictionCount = 5;
    if (PNeuralNet->trainning.pPredictions == NULL)
    {
        PNeuralNet->trainning.pPredictions = malloc(sizeof(TPPrediction) * defaultPredictionCount);
        if (PNeuralNet->trainning.pPredictions == NULL)
            return;

        for (uint16_t i = 0; i < PNeuralNet->trainning.predictionCount; i++)
        {
            PNeuralNet->trainning.pPredictions[i] = NULL;
        }
    }

    if (PNeuralNet->trainning.predictionCount > 0)
    {
        for (uint16_t i = 0; i < PNeuralNet->trainning.predictionCount; i++)
        {
            free(PNeuralNet->trainning.pPredictions[i]);
        }
        PNeuralNet->trainning.predictionCount = 0;
    }

    if (PNeuralNet->trainning.pPredictions == NULL)
        return;
    TPSoftmaxLayer PSoftmaxLayer = ((TPSoftmaxLayer)PNeuralNet->layers[PNeuralNet->depth - 1]);
    for (uint32_t i = 0; i < PSoftmaxLayer->layer.out_v->weight->length; i++)
    {
        TPrediction* pPrediction = NULL;

        if (PNeuralNet->trainning.pPredictions[PNeuralNet->trainning.predictionCount] == NULL)
            pPrediction = malloc(sizeof(TPrediction));
        else
            pPrediction = PNeuralNet->trainning.pPredictions[PNeuralNet->trainning.predictionCount];

        if (pPrediction == NULL)
            break;
        pPrediction->labelIndex = i;
        pPrediction->likeliHood = PSoftmaxLayer->layer.out_v->weight->buffer[i];

        if (PNeuralNet->trainning.predictionCount < defaultPredictionCount)
        {
            PNeuralNet->trainning.pPredictions[PNeuralNet->trainning.predictionCount] = pPrediction;
            PNeuralNet->trainning.predictionCount++;
        }
        else
        {
            for (uint16_t i = 0; i < defaultPredictionCount; i++)
            {
                if (pPrediction->likeliHood > PNeuralNet->trainning.pPredictions[i]->likeliHood)
                {
                    PNeuralNet->trainning.pPredictions[i] = pPrediction;
                    break;
                }
            }
        }
    }
}

void NeuralNetPrintWeights(TPNeuralNet PNeuralNet, uint16_t LayerIndex, uint8_t InOut)
{
    // float32_t maxv = 0;
    // uint16_t maxi = 0;
    TPLayer pNetLayer = (TPLayer)(PNeuralNet->layers[LayerIndex]);
    // for (uint16_t i = 0; i < pNetLayer->out_v->weight->length; i++)
    //{
    //	LOGINFOR("LayerType=%s PNeuralNet->depth=%d weight=%f", CNNTypeName[pNetLayer->LayerType], PNeuralNet->depth - 1, pNetLayer->out_v->weight->buffer[i]);
    // }
    if (InOut == 0)
    {
        LOGINFOR("layers[%d] out_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
        if (pNetLayer->out_v != NULL)
            pNetLayer->out_v->print(pNetLayer->out_v, PRINTFLAG_WEIGHT);
        else
            LOGINFOR("pNetLayer->out_v=NULL");
    }
    else if (InOut == 1)
    {
        LOGINFOR("layers[%d] in_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth);
        if (pNetLayer->in_v != NULL)
            pNetLayer->in_v->print(pNetLayer->in_v, PRINTFLAG_WEIGHT);
        else
            LOGINFOR("pNetLayer->in_v=NULL");
    }
    else if (pNetLayer->LayerType == Layer_Type_Convolution)
    {
        for (uint16_t i = 0; i < ((TPConvLayer)pNetLayer)->filters->filterNumber; i++)
        {
            LOGINFOR("layers[%d] type=%s w=%d h=%d depth=%d filterNumber=%d/%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], ((TPConvLayer)pNetLayer)->filters->_w, ((TPConvLayer)pNetLayer)->filters->_h, ((TPConvLayer)pNetLayer)->filters->_depth, i, ((TPConvLayer)pNetLayer)->filters->filterNumber);
            ((TPConvLayer)pNetLayer)->filters->volumes[i]->print(((TPConvLayer)pNetLayer)->filters->volumes[i], PRINTFLAG_WEIGHT);
        }
        LOGINFOR("biases:w=%d h=%d depth=%d", ((TPConvLayer)pNetLayer)->biases->_w, ((TPConvLayer)pNetLayer)->biases->_h, ((TPConvLayer)pNetLayer)->biases->_depth);
        ((TPConvLayer)pNetLayer)->biases->print(((TPConvLayer)pNetLayer)->biases, PRINTFLAG_WEIGHT);
    }
    else if (pNetLayer->LayerType == Layer_Type_FullyConnection)
    {
        for (uint16_t i = 0; i < ((TPFullyConnLayer)pNetLayer)->filters->filterNumber; i++)
        {
            LOGINFOR("layers[%d] type=%s w=%d h=%d depth=%d filterNumber=%d/%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], ((TPConvLayer)pNetLayer)->filters->_w, ((TPConvLayer)pNetLayer)->filters->_h, ((TPConvLayer)pNetLayer)->filters->_depth, i, ((TPConvLayer)pNetLayer)->filters->filterNumber);
            ((TPFullyConnLayer)pNetLayer)->filters->volumes[i]->print(((TPFullyConnLayer)pNetLayer)->filters->volumes[i], PRINTFLAG_WEIGHT);
        }
        LOGINFOR("biases:w=%d h=%d depth=%d", ((TPFullyConnLayer)pNetLayer)->biases->_w, ((TPFullyConnLayer)pNetLayer)->biases->_h, ((TPFullyConnLayer)pNetLayer)->biases->_depth);
        ((TPFullyConnLayer)pNetLayer)->biases->print(((TPFullyConnLayer)pNetLayer)->biases, PRINTFLAG_WEIGHT);
    }
    else if (pNetLayer->LayerType == Layer_Type_SoftMax)
    {
        PNeuralNet->printTensor("EXP", ((TPSoftmaxLayer)pNetLayer)->exp);
    }
    else if (pNetLayer->LayerType == Layer_Type_Pool)
    {
        //((TPPoolLayer)pNetLayer)->filter->print(((TPPoolLayer)pNetLayer)->filter, PRINTFLAG_WEIGHT);
    }
}
/// <summary>
/// discard
/// </summary>
/// <param name="PNeuralNet"></param>
/// <param name="LayerIndex"></param>
/// <param name="InOut"></param>
void NeuralNetPrintFilters(TPNeuralNet PNeuralNet, uint16_t LayerIndex, uint8_t InOut)
{
    TPLayer pNetLayer = (PNeuralNet->layers[LayerIndex]);

    if (InOut == 0)
    {
        LOGINFOR("layers[%d] out_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
        pNetLayer->out_v->print(pNetLayer->out_v, PRINTFLAG_WEIGHT);
    }
    else if (InOut == 1)
    {
        LOGINFOR("layers[%d] in_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
        pNetLayer->in_v->print(pNetLayer->in_v, PRINTFLAG_WEIGHT);
    }
    else if (pNetLayer->LayerType == Layer_Type_Convolution)
    {
        LOGINFOR("layers[%d] type=%s w=%d h=%d depth=%d filterNumber=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, ((TPConvLayer)pNetLayer)->filters->filterNumber);
        for (uint16_t i = 0; i < ((TPConvLayer)pNetLayer)->filters->filterNumber; i++)
        {
            ((TPConvLayer)pNetLayer)->filters->volumes[i]->print(((TPConvLayer)pNetLayer)->filters->volumes[i], PRINTFLAG_WEIGHT);
        }
        ((TPConvLayer)pNetLayer)->biases->print(((TPConvLayer)pNetLayer)->biases, PRINTFLAG_WEIGHT);
    }
    else if (pNetLayer->LayerType == Layer_Type_FullyConnection)
    {
        LOGINFOR("layers[%d] type=%s w=%d h=%d depth=%d filterNumber=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, ((TPConvLayer)pNetLayer)->filters->filterNumber);
        for (uint16_t i = 0; i < ((TPFullyConnLayer)pNetLayer)->filters->filterNumber; i++)
        {
            ((TPFullyConnLayer)pNetLayer)->filters->volumes[i]->print(((TPFullyConnLayer)pNetLayer)->filters->volumes[i], PRINTFLAG_WEIGHT);
        }
        ((TPFullyConnLayer)pNetLayer)->biases->print(((TPFullyConnLayer)pNetLayer)->biases, PRINTFLAG_WEIGHT);
    }
}

void NeuralNetPrintGradients(TPNeuralNet PNeuralNet, uint16_t LayerIndex, uint8_t InOut)
{
    // float32_t maxv = 0;
    // uint16_t maxi = 0;
    TPLayer pNetLayer = ((TPLayer)PNeuralNet->layers[LayerIndex]);
    // for (uint16_t i = 0; i < pNetLayer->out_v->weight->length; i++)
    //{
    //	//LOGINFOR("LayerType=%s PNeuralNet->depth=%d weight_grad=%f", CNNTypeName[pNetLayer->LayerType], PNeuralNet->depth - 1, pNetLayer->out_v->weight_grad->buffer[i]);
    // }
    if (InOut == 0)
    {
        LOGINFOR("layers[%d] out_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
        pNetLayer->out_v->print(pNetLayer->out_v, PRINTFLAG_GRADS);
    }
    else if (InOut == 1)
    {
        LOGINFOR("layers[%d] in_v type=%s w=%d h=%d depth=%d", LayerIndex, CNNTypeName[pNetLayer->LayerType], pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth);
        if (pNetLayer->in_v != NULL)
            pNetLayer->in_v->print(pNetLayer->in_v, PRINTFLAG_GRADS);
    }
    else if (pNetLayer->LayerType == Layer_Type_Convolution)
    {
        LOGINFOR("layers[%d] w=%d h=%d depth=%d filterNumber=%d %s", LayerIndex, pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, ((TPConvLayer)pNetLayer)->filters->filterNumber, CNNTypeName[pNetLayer->LayerType]);
        for (uint16_t i = 0; i < ((TPConvLayer)pNetLayer)->filters->filterNumber; i++)
        {
            ((TPConvLayer)pNetLayer)->filters->volumes[i]->print(((TPConvLayer)pNetLayer)->filters->volumes[i], PRINTFLAG_GRADS);
        }
    }
    else if (pNetLayer->LayerType == Layer_Type_FullyConnection)
    {
        LOGINFOR("layers[%d] w=%d h=%d depth=%d filterNumber=%d %s", LayerIndex, pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, ((TPConvLayer)pNetLayer)->filters->filterNumber, CNNTypeName[pNetLayer->LayerType]);
        for (uint16_t i = 0; i < ((TPFullyConnLayer)pNetLayer)->filters->filterNumber; i++)
        {
            ((TPFullyConnLayer)pNetLayer)->filters->volumes[i]->print(((TPFullyConnLayer)pNetLayer)->filters->volumes[i], PRINTFLAG_GRADS);
        }
    }
}

void NeuralNetPrint(char* Name, TPTensor PTensor)
{
    for (uint16_t i = 0; i < PTensor->length; i++)
    {
        if (i % 16 == 0)
            LOG("\n");
        LOG("%s[%d]=" PRINTFLAG_FORMAT, Name, i, PTensor->buffer[i]);
    }
    LOG("\n");
}

void NeuralNetPrintLayersInfor(TPNeuralNet PNeuralNet)
{
    TPLayer pNetLayer;
    for (uint16_t out_d = 0; out_d < PNeuralNet->depth; out_d++)
    {
        pNetLayer = PNeuralNet->layers[out_d];
        if (pNetLayer->LayerType == Layer_Type_Convolution)
        {
            LOG("[NeuralNetLayerInfor[%02d,%02d]]:in_w=%2d in_h=%2d in_depth=%2d out_w=%2d out_h=%2d out_depth=%2d %-15s fileterNumber=%d size=%dx%dx%d\n", out_d, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
                pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, PNeuralNet->getName(pNetLayer->LayerType), ((TPConvLayer)pNetLayer)->filters->filterNumber,
                ((TPConvLayer)pNetLayer)->filters->_w, ((TPConvLayer)pNetLayer)->filters->_h, ((TPConvLayer)pNetLayer)->filters->_depth);
        }
        else if (pNetLayer->LayerType == Layer_Type_FullyConnection)
        {
            LOG("[NeuralNetLayerInfor[%02d,%02d]]:in_w=%2d in_h=%2d in_depth=%2d out_w=%2d out_h=%2d out_depth=%2d %-15s fileterNumber=%d size=%dx%dx%d\n", out_d, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
                pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, PNeuralNet->getName(pNetLayer->LayerType), ((TPConvLayer)pNetLayer)->filters->filterNumber,
                ((TPFullyConnLayer)pNetLayer)->filters->_w, ((TPFullyConnLayer)pNetLayer)->filters->_h, ((TPFullyConnLayer)pNetLayer)->filters->_depth);
        }
        else if (pNetLayer->LayerType == Layer_Type_SoftMax)
        {
            LOG("[NeuralNetLayerInfor[%02d,%02d]]:in_w=%2d in_h=%2d in_depth=%2d out_w=%2d out_h=%2d out_depth=%2d %s\n", out_d, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
                pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, PNeuralNet->getName(pNetLayer->LayerType));
        }
        else
        {
            LOG("[NeuralNetLayerInfor[%02d,%02d]]:in_w=%2d in_h=%2d in_depth=%2d out_w=%2d out_h=%2d out_depth=%2d %s\n", out_d, pNetLayer->LayerType, pNetLayer->in_w, pNetLayer->in_h, pNetLayer->in_depth,
                pNetLayer->out_w, pNetLayer->out_h, pNetLayer->out_depth, PNeuralNet->getName(pNetLayer->LayerType));
        }
    }
}
void NeuralNetPrintTrainningInfor(TPNeuralNet PNeuralNet)
{
    time_t avg_iterations_time = 0;
#if 1
    LOGINFOR("DatasetTotal    :%09ld  ", PNeuralNet->trainning.datasetTotal);
    LOGINFOR("DatasetIndex    :%09ld  ", PNeuralNet->trainning.trinning_dataset_index);
    LOGINFOR("EpochCount      :%09ld  ", PNeuralNet->trainning.epochCount);
    LOGINFOR("SampleCount     :%09ld  ", PNeuralNet->trainning.sampleCount);
    LOGINFOR("LabelIndex      :%09ld  ", PNeuralNet->trainning.labelIndex);
    LOGINFOR("BatchCount      :%09ld  ", PNeuralNet->trainning.batchCount);
    LOGINFOR("Iterations      :%09ld  ", PNeuralNet->trainning.iterations);

    LOGINFOR("AverageCostLoss :%.6f", PNeuralNet->trainning.sum_cost_loss / PNeuralNet->trainning.sampleCount);
    LOGINFOR("L1_decay_loss   :%.6f", PNeuralNet->trainning.sum_l1_decay_loss / PNeuralNet->trainning.sampleCount);
    LOGINFOR("L2_decay_loss   :%.6f", PNeuralNet->trainning.sum_l2_decay_loss / PNeuralNet->trainning.sampleCount);
    LOGINFOR("TrainingAccuracy:%.6f", PNeuralNet->trainning.trainingAccuracy / PNeuralNet->trainning.sampleCount);
    LOGINFOR("TestingAccuracy :%.6f", PNeuralNet->trainning.testingAccuracy / PNeuralNet->trainning.sampleCount);

    if (PNeuralNet->trainning.iterations > 0)
        avg_iterations_time = PNeuralNet->totalTime / PNeuralNet->trainning.iterations;

    LOGINFOR("TotalElapsedTime:%9lld", PNeuralNet->totalTime);
    LOGINFOR("ForwardTime     :%09lld", PNeuralNet->fwTime);
    LOGINFOR("BackwardTime    :%09lld", PNeuralNet->bwTime);
    LOGINFOR("OptimTime       :%09lld", PNeuralNet->optimTime);
    LOGINFOR("AvgBatchTime    :%09lld", avg_iterations_time);
    LOGINFOR("AvgSampleTime   :%09lld", PNeuralNet->totalTime / PNeuralNet->trainning.sampleCount);
#else
    LOGINFOR("DatasetTotal:%06d DatasetIndex:%06d EpochCount:%06d SampleCount:%06d LabelIndex:%06d BatchCount:%06d Iterations:%06d",
             PNeuralNet->trainning.datasetTotal,
             PNeuralNet->trainning.trinning_dataset_index,
             PNeuralNet->trainning.epochCount,
             PNeuralNet->trainning.sampleCount,
             PNeuralNet->trainning.labelIndex,
             PNeuralNet->trainning.batchCount,
             PNeuralNet->trainning.iterations);

    LOGINFOR("AvgCostLoss:%.6f L1_decay_loss:%.6f L2_decay_loss:%.6f TrainingAccuracy:%.6f TestingAccuracy:%.6f",
             PNeuralNet->trainning.cost_loss_sum / PNeuralNet->trainning.sampleCount,
             PNeuralNet->trainning.l1_decay_loss / PNeuralNet->trainning.sampleCount,
             PNeuralNet->trainning.l2_decay_loss / PNeuralNet->trainning.sampleCount,
             PNeuralNet->trainning.trainingAccuracy / PNeuralNet->trainning.sampleCount,
             PNeuralNet->trainning.testingAccuracy / PNeuralNet->trainning.sampleCount);

    if (PNeuralNet->trainning.iterations > 0)
        avg_iterations_time = PNeuralNet->totalTime / PNeuralNet->trainning.iterations;

    LOGINFOR("TotalTime:%lld ForwardTime:%05lld BackwardTime:%05lld OptimTime:%05lld AvgBatchTime:%05lld AvgSampleTime:%05lld",
             PNeuralNet->totalTime,
             PNeuralNet->fwTime,
             PNeuralNet->bwTime,
             PNeuralNet->optimTime,
             avg_iterations_time,
             PNeuralNet->totalTime / PNeuralNet->trainning.sampleCount);
#endif
}

void NeuralNetTrain(TPNeuralNet PNeuralNet, TPVolume PVolume)
{
    TPTensor weight, grads;
    TPTensor accum_grads1; // 累计历史梯度
    TPTensor accum_grads2; // for Optm_Adadelta
    TPParameters* pResponseResults = NULL;
    TPParameters pResponse = NULL;

    float32_t l1_decay = 0.00;
    float32_t l2_decay = 0.00;
    float32_t l1_decay_grad = 0.00;
    float32_t l2_decay_grad = 0.00;
    float32_t bias1 = 0.00;
    float32_t bias2 = 0.00;

    float32_t delta_x = 0.00;
    float32_t gradij = 0.00;
    float32_t cost_loss = 0.00;
    float32_t l1_decay_loss = 0.00;
    float32_t l2_decay_loss = 0.00;

    accum_grads1 = NULL;
    accum_grads2 = NULL;
    time_t starTick = 0;
    /////////////////////////////////////////////////////////////////////////////////////
    PNeuralNet->optimTime = 0;
    starTick = GetTimestamp();
    PNeuralNet->forward(PNeuralNet, PVolume);
    PNeuralNet->getCostLoss(PNeuralNet, &cost_loss);
    PNeuralNet->fwTime = GetTimestamp() - starTick;
    starTick = GetTimestamp();
    PNeuralNet->backward(PNeuralNet);
    PNeuralNet->bwTime = GetTimestamp() - starTick;
    PNeuralNet->trainning.batchCount++;
    PNeuralNet->trainning.sum_cost_loss = PNeuralNet->trainning.sum_cost_loss + cost_loss;

    /////////////////////////////////////////////////////////////////////////////////////
    // 小批量阀值batch_size
    if (PNeuralNet->trainningParam.batch_size <= 0)
        PNeuralNet->trainningParam.batch_size = 15;
    if (PNeuralNet->trainning.batchCount % PNeuralNet->trainningParam.batch_size != 0)
        return;
    /////////////////////////////////////////////////////////////////////////////////////
    ////小批量梯度下降
    starTick = GetTimestamp();
    PNeuralNet->trainning.iterations++;
    PNeuralNet->getWeightsAndGrads(PNeuralNet);
    if (PNeuralNet->trainning.underflow)
    {
        PNeuralNet->trainning.trainningGoing = false;
        LOGERROR("PNeuralNet->trainning.underflow = true");
        return;
    }
    if (PNeuralNet->trainning.overflow)
    {
        PNeuralNet->trainning.trainningGoing = false;
        LOGERROR("PNeuralNet->trainning.overflow = true");
        return;
    }
    ///////////////////////////////////////////////////////////////////
    // LOGERROR("Debug:Stop");
    // PNeuralNet->trainning.trainningGoing = false;
    // return;
    //////////////////////////////////////////////////////////////////
    pResponseResults = PNeuralNet->trainning.pResponseResults;
    if (pResponseResults == NULL)
    {
        LOGERROR("No ResponseResults! pResponseResults=NULL");
        PNeuralNet->trainning.trainningGoing = false;
        return;
    }
    if (PNeuralNet->trainning.responseCount <= 0)
    {
        LOGERROR("No ResponseResults! trainning.responseCount <= 0");
        PNeuralNet->trainning.trainningGoing = false;
        return;
    }
    if (PNeuralNet->trainning.grads_sum1 == NULL || PNeuralNet->trainning.grads_sum_count <= 0)
    {
        PNeuralNet->trainning.grads_sum1 = malloc(sizeof(TPTensor) * PNeuralNet->trainning.responseCount);
        PNeuralNet->trainning.grads_sum2 = malloc(sizeof(TPTensor) * PNeuralNet->trainning.responseCount);
        PNeuralNet->trainning.grads_sum_count = PNeuralNet->trainning.responseCount;
        if (PNeuralNet->trainning.grads_sum1 == NULL || PNeuralNet->trainning.grads_sum2 == NULL)
        {
            LOGERROR("No trainning.grads_sum1 or trainning.grads_sum2 NULL");
            PNeuralNet->trainning.trainningGoing = false;
            return;
        }
        for (uint16_t i = 0; i < PNeuralNet->trainning.responseCount; i++)
        {
            PNeuralNet->trainning.grads_sum1[i] = MakeTensor(pResponseResults[i]->filterWeight->length);
            PNeuralNet->trainning.grads_sum2[i] = MakeTensor(pResponseResults[i]->filterWeight->length);
        }
    }
    /////////////////////////////////////////////////////////////////////////////////////
    /// 计算更新整个网络层的权重参数
    for (uint16_t i = 0; i < PNeuralNet->trainning.responseCount; i++)
    {
        pResponse = pResponseResults[i];
        weight = pResponse->filterWeight;
        grads = pResponse->filterGrads;
        accum_grads1 = PNeuralNet->trainning.grads_sum1[i];
        accum_grads2 = PNeuralNet->trainning.grads_sum2[i];
        l1_decay = PNeuralNet->trainningParam.l1_decay_rate * pResponse->l1_decay_rate;
        l2_decay = PNeuralNet->trainningParam.l2_decay_rate * pResponse->l2_decay_rate;

        if (weight->length <= 0 || grads->length <= 0 || accum_grads1 == NULL || accum_grads2 == NULL)
        {
            LOGERROR("response weight->length=%d grads->length=%d grads_sum_count=%d", weight->length, grads->length, PNeuralNet->trainning.grads_sum_count);
            PNeuralNet->trainning.trainningGoing = false;
            break;
        }

        for (uint16_t j = 0; j < weight->length; j++) // update all weight parameters
        {
            l1_decay_loss = l1_decay_loss + l1_decay * abs(weight->buffer[j]);
            l2_decay_loss = l2_decay_loss + l2_decay * weight->buffer[j] * weight->buffer[j] / 2;

            l1_decay_grad = (weight->buffer[j] > 0) ? l1_decay : -l1_decay;
            l2_decay_grad = l2_decay * weight->buffer[j]; // 函数 y = ax^2 的导数是 dy/dx = 2ax

            gradij = (grads->buffer[j] + l1_decay_grad + l2_decay_grad) / PNeuralNet->trainningParam.batch_size;

            switch (PNeuralNet->trainningParam.optimize_method)
            {
            case Optm_Adam:
                accum_grads1->buffer[j] = accum_grads1->buffer[j] * PNeuralNet->trainningParam.beta1 + (1 - PNeuralNet->trainningParam.beta1) * gradij;
                accum_grads2->buffer[j] = accum_grads1->buffer[j] * PNeuralNet->trainningParam.beta2 + (1 - PNeuralNet->trainningParam.beta2) * gradij * gradij;
                bias1 = accum_grads1->buffer[j] * (1 - pow(PNeuralNet->trainningParam.beta1, PNeuralNet->trainning.batchCount));
                bias2 = accum_grads1->buffer[j] * (1 - pow(PNeuralNet->trainningParam.beta2, PNeuralNet->trainning.batchCount));
                delta_x = -PNeuralNet->trainningParam.learning_rate * bias1 / (sqrt(bias2) + PNeuralNet->trainningParam.eps);
                weight->buffer[j] = weight->buffer[j] + delta_x;
                break;
            case Optm_Adagrad:
                accum_grads1->buffer[j] = accum_grads1->buffer[j] + gradij * gradij;
                delta_x = -PNeuralNet->trainningParam.learning_rate / sqrt(accum_grads1->buffer[j] + PNeuralNet->trainningParam.eps) * gradij;
                weight->buffer[j] = weight->buffer[j] + delta_x;
                break;
            case Optm_Adadelta:
                accum_grads1->buffer[j] = accum_grads1->buffer[j] * PNeuralNet->trainningParam.momentum + (1 - PNeuralNet->trainningParam.momentum) * gradij * gradij;
                delta_x = -sqrt((accum_grads2->buffer[j] + PNeuralNet->trainningParam.eps) / (accum_grads1->buffer[j] + PNeuralNet->trainningParam.eps)) * gradij;
                accum_grads2->buffer[j] = accum_grads2->buffer[j] * PNeuralNet->trainningParam.momentum + (1 - PNeuralNet->trainningParam.momentum) * delta_x * delta_x;
                weight->buffer[j] = weight->buffer[j] + delta_x;
                break;
            default:
                break;
            } // switch
            grads->buffer[j] = 0;
        }
    }
    for (uint16_t i = 0; i < PNeuralNet->trainning.responseCount; i++)
    {
        pResponse = pResponseResults[i];
        free(pResponse);
    }
    free(PNeuralNet->trainning.pResponseResults);
    PNeuralNet->trainning.responseCount = 0;
    PNeuralNet->trainning.pResponseResults = NULL;
    PNeuralNet->optimTime = GetTimestamp() - starTick;

    PNeuralNet->trainning.sum_l1_decay_loss = (PNeuralNet->trainning.sum_l1_decay_loss + l1_decay_loss);
    PNeuralNet->trainning.sum_l2_decay_loss = (PNeuralNet->trainning.sum_l2_decay_loss + l2_decay_loss);
    // PNeuralNet->trainning.cost_loss_sum = PNeuralNet->trainning.cost_loss_sum + cost_loss;
}

void NeuralNetPredict(TPNeuralNet PNeuralNet, TPVolume PVolume)
{
    float32_t cost_loss = 0;
    PNeuralNet->forward(PNeuralNet, PVolume);
    PNeuralNet->getCostLoss(PNeuralNet, &cost_loss);
}
/// @brief ///////////////////////////////////////////////////////////////////////
/// @param PNeuralNet
void NeuralNetSaveWeights(TPNeuralNet PNeuralNet)
{
    FILE* pFile = NULL;
    char* name = NULL;
    if (PNeuralNet == NULL)
        return;

    if (PNeuralNet->name != NULL)
    {
        name = (char*)malloc(strlen(PNeuralNet->name) + strlen(NEURALNET_CNN_WEIGHT_FILE_NAME) + 3);
        sprintf(name, "%s_%02d%s", PNeuralNet->name, PNeuralNet->depth, NEURALNET_CNN_WEIGHT_FILE_NAME);
        pFile = fopen(name, "wb");
    }
    else
    {
        pFile = fopen(NEURALNET_CNN_WEIGHT_FILE_NAME, "wb");
    }

    if (pFile != NULL)
    {
        for (uint16_t layerIndex = 0; layerIndex < PNeuralNet->depth; layerIndex++)
        {
            TPLayer pNetLayer = (PNeuralNet->layers[layerIndex]);
            switch (pNetLayer->LayerType)
            {
            case Layer_Type_Input:
                break;
            case Layer_Type_Convolution:
            {
                for (uint16_t out_d = 0; out_d < ((TPConvLayer)pNetLayer)->filters->filterNumber; out_d++)
                {
                    TensorSave(pFile, ((TPConvLayer)pNetLayer)->filters->volumes[out_d]->weight);
                }
                TensorSave(pFile, ((TPConvLayer)pNetLayer)->biases->weight);
                break;
            }
            case Layer_Type_Pool:
                break;
            case Layer_Type_ReLu:
                break;
            case Layer_Type_FullyConnection:
            {
                for (uint16_t out_d = 0; out_d < ((TPFullyConnLayer)pNetLayer)->filters->filterNumber; out_d++)
                {
                    TensorSave(pFile, ((TPFullyConnLayer)pNetLayer)->filters->volumes[out_d]->weight);
                }
                TensorSave(pFile, ((TPFullyConnLayer)pNetLayer)->biases->weight);
                break;
            }
            case Layer_Type_SoftMax:
                break;
            default:
                break;
            }
        }
        fclose(pFile);
        LOGINFOR("save weights to file %s", name);
    }
    // if (name != NULL)
    //	free(name);
}

void NeuralNetLoadWeights(TPNeuralNet PNeuralNet)
{
    FILE* pFile = NULL;
    char* name = NULL;
    if (PNeuralNet == NULL)
        return;
    if (PNeuralNet->name != NULL)
    {
        name = (char*)malloc(strlen(PNeuralNet->name) + strlen(NEURALNET_CNN_WEIGHT_FILE_NAME) + 3);
        sprintf(name, "%s_%02d%s", PNeuralNet->name, PNeuralNet->depth, NEURALNET_CNN_WEIGHT_FILE_NAME);
        pFile = fopen(name, "rb");
    }
    else
    {
        pFile = fopen(NEURALNET_CNN_WEIGHT_FILE_NAME, "rb");
        name = NEURALNET_CNN_WEIGHT_FILE_NAME;
    }

    if (pFile != NULL)
    {
        for (uint16_t layerIndex = 0; layerIndex < PNeuralNet->depth; layerIndex++)
        {
            TPLayer pNetLayer = (PNeuralNet->layers[layerIndex]);
            switch (pNetLayer->LayerType)
            {
            case Layer_Type_Input:
                break;
            case Layer_Type_Convolution:
            {
                for (uint16_t out_d = 0; out_d < ((TPConvLayer)pNetLayer)->filters->filterNumber; out_d++)
                {
                    TensorLoad(pFile, ((TPConvLayer)pNetLayer)->filters->volumes[out_d]->weight);
                }
                TensorLoad(pFile, ((TPConvLayer)pNetLayer)->biases->weight);
                break;
            }
            case Layer_Type_ReLu:
                break;
            case Layer_Type_Pool:
                break;
            case Layer_Type_FullyConnection:
            {
                for (uint16_t out_d = 0; out_d < ((TPFullyConnLayer)pNetLayer)->filters->filterNumber; out_d++)
                {
                    TensorLoad(pFile, ((TPFullyConnLayer)pNetLayer)->filters->volumes[out_d]->weight);
                }
                TensorLoad(pFile, ((TPFullyConnLayer)pNetLayer)->biases->weight);
                break;
            }
            case Layer_Type_SoftMax:
                break;
            default:
                break;
            }
        }
        LOGINFOR("Loaded weights from file %s", name);
        fclose(pFile);
    }
    else
    {
        LOGINFOR("Loaded weights from file %s failed file not found!", name);
    }
    // if (name != NULL)
    //	free(name);
}

void NeuralNetSaveNet(TPNeuralNet PNeuralNet)
{
    FILE* pFile = NULL;
    char* name = NULL;
    if (PNeuralNet == NULL)
        return;
    if (PNeuralNet->name != NULL)
    {
        name = (char*)malloc(strlen(PNeuralNet->name) + strlen(NEURALNET_CNN_FILE_NAME));
        sprintf(name, "%s%s", PNeuralNet->name, NEURALNET_CNN_FILE_NAME);
        pFile = fopen(name, "wb");
    }
    else
    {
        pFile = fopen(NEURALNET_CNN_FILE_NAME, "wb");
    }
    if (pFile != NULL)
    {
        for (uint16_t layerIndex = 0; layerIndex < PNeuralNet->depth; layerIndex++)
        {
            TPLayer pNetLayer = (PNeuralNet->layers[layerIndex]);
            switch (pNetLayer->LayerType)
            {
            case Layer_Type_Input:
                break;
            case Layer_Type_Convolution:
                break;
            case Layer_Type_ReLu:
                break;
            case Layer_Type_Pool:
                break;
            case Layer_Type_FullyConnection:
                break;
            case Layer_Type_SoftMax:
                break;
            default:
                break;
            }
        }
        fclose(pFile);
        LOGINFOR("save weights to file %s", name);
    }
}

void NeuralNetLoadNet(TPNeuralNet PNeuralNet)
{
}

TPNeuralNet NeuralNetCNNCreate(char* name)
{
    TPNeuralNet PNeuralNet = malloc(sizeof(TNeuralNet));

    if (PNeuralNet == NULL)
    {
        LOGERROR("PNeuralNet==NULL!");
        return NULL;
    }
    PNeuralNet->name = name;
    PNeuralNet->init = NeuralNetInit;
    PNeuralNet->free = NeuralNetFree;
    PNeuralNet->forward = NeuralNetForward;
    PNeuralNet->backward = NeuralNetBackward;
    PNeuralNet->getWeightsAndGrads = NeuralNetGetWeightsAndGrads;
    PNeuralNet->getCostLoss = NeuralNetComputeCostLoss;
    PNeuralNet->getPredictions = NeuralNetUpdatePrediction;
    PNeuralNet->getMaxPrediction = NeuralNetGetMaxPrediction;
    PNeuralNet->train = NeuralNetTrain;
    PNeuralNet->predict = NeuralNetPredict;
    PNeuralNet->saveWeights = NeuralNetSaveWeights;
    PNeuralNet->loadWeights = NeuralNetLoadWeights;
    PNeuralNet->printGradients = NeuralNetPrintGradients;
    PNeuralNet->printWeights = NeuralNetPrintWeights;
    PNeuralNet->printTrainningInfor = NeuralNetPrintTrainningInfor;
    PNeuralNet->printNetLayersInfor = NeuralNetPrintLayersInfor;
    PNeuralNet->printTensor = NeuralNetPrint;
    PNeuralNet->getName = NeuralNetGetLayerName;

    return PNeuralNet;
}

char* NeuralNetGetLayerName(TLayerType LayerType)
{
    return CNNTypeName[LayerType];
}
int NeuralNetAddLayer(TPNeuralNet PNeuralNet, TLayerOption LayerOption)
{
    TPLayer pNetLayer = NULL;
    if (PNeuralNet == NULL)
        return NEURALNET_ERROR_BASE;
    switch (LayerOption.LayerType)
    {
    case Layer_Type_Input:
        // LayerOption.in_w = LayerOption.in_w;
        // LayerOption.in_h = LayerOption.in_h;
        // LayerOption.in_depth = LayerOption.in_depth;
        if (PNeuralNet->depth > 0)
        {
            LOGINFOR("input layer need to add first");
            return LayerOption.LayerType + NEURALNET_ERROR_BASE;
        }
        PNeuralNet->init(PNeuralNet, &LayerOption);
        break;
    case Layer_Type_Convolution:
    {
        if (PNeuralNet->depth <= 0)
            return LayerOption.LayerType + NEURALNET_ERROR_BASE;
        pNetLayer = PNeuralNet->layers[PNeuralNet->depth - 1];
        LayerOption.LayerType = Layer_Type_Convolution;
        LayerOption.in_w = pNetLayer->out_w;
        LayerOption.in_h = pNetLayer->out_h;
        LayerOption.in_depth = pNetLayer->out_depth;
        LayerOption.filter_w = LayerOption.filter_w;
        LayerOption.filter_h = LayerOption.filter_h;
        LayerOption.filter_depth = LayerOption.in_depth;
        LayerOption.filter_number = LayerOption.filter_number;
        LayerOption.stride = LayerOption.stride;
        LayerOption.padding = LayerOption.padding;
        LayerOption.bias = LayerOption.bias;
        LayerOption.l1_decay_rate = LayerOption.l1_decay_rate;
        LayerOption.l2_decay_rate = LayerOption.l2_decay_rate;
        PNeuralNet->init(PNeuralNet, &LayerOption);
        return LayerOption.LayerType;
        break;
    }
    case Layer_Type_ReLu:
        if (PNeuralNet->depth <= 0)
            return LayerOption.LayerType + NEURALNET_ERROR_BASE;
        pNetLayer = PNeuralNet->layers[PNeuralNet->depth - 1];
        LayerOption.in_w = pNetLayer->out_w;
        LayerOption.in_h = pNetLayer->out_h;
        LayerOption.in_depth = pNetLayer->out_depth;
        PNeuralNet->init(PNeuralNet, &LayerOption);
        return LayerOption.LayerType;
        break;
    case Layer_Type_Pool:
        if (PNeuralNet->depth <= 0)
            return LayerOption.LayerType + NEURALNET_ERROR_BASE;
        pNetLayer = PNeuralNet->layers[PNeuralNet->depth - 1];
        LayerOption.LayerType = Layer_Type_Pool;
        LayerOption.in_w = pNetLayer->out_w;
        LayerOption.in_h = pNetLayer->out_h;
        LayerOption.in_depth = pNetLayer->out_depth;
        LayerOption.filter_w = LayerOption.filter_w;
        LayerOption.filter_h = LayerOption.filter_h;
        LayerOption.filter_depth = LayerOption.in_depth;
        PNeuralNet->init(PNeuralNet, &LayerOption);
        return LayerOption.LayerType;
        break;
    case Layer_Type_FullyConnection:
    {
        if (PNeuralNet->depth <= 0)
            return LayerOption.LayerType + NEURALNET_ERROR_BASE;
        pNetLayer = PNeuralNet->layers[PNeuralNet->depth - 1];
        LayerOption.in_w = pNetLayer->out_w;
        LayerOption.in_h = pNetLayer->out_h;
        LayerOption.in_depth = pNetLayer->out_depth;

        LayerOption.filter_w = LayerOption.filter_w;
        LayerOption.filter_h = LayerOption.filter_h;
        LayerOption.filter_depth = LayerOption.in_w * LayerOption.in_h * LayerOption.in_depth;
        LayerOption.filter_number = LayerOption.filter_number;
        LayerOption.out_depth = LayerOption.filter_number;
        LayerOption.out_h = 1;
        LayerOption.out_w = 1;
        LayerOption.bias = LayerOption.bias;
        LayerOption.l1_decay_rate = LayerOption.l1_decay_rate;
        LayerOption.l2_decay_rate = LayerOption.l2_decay_rate;
        PNeuralNet->init(PNeuralNet, &LayerOption);
        return LayerOption.LayerType;
        break;
    }
    case Layer_Type_SoftMax:
        if (PNeuralNet->depth <= 0)
            return LayerOption.LayerType + NEURALNET_ERROR_BASE;
        pNetLayer = PNeuralNet->layers[PNeuralNet->depth - 1];
        LayerOption.in_w = pNetLayer->out_w;
        LayerOption.in_h = pNetLayer->out_h;
        LayerOption.in_depth = pNetLayer->out_depth;

        LayerOption.out_h = 1;
        LayerOption.out_w = 1;
        LayerOption.out_depth = LayerOption.in_depth * LayerOption.in_w * LayerOption.in_h;
        PNeuralNet->init(PNeuralNet, &LayerOption);
        return LayerOption.LayerType;
        break;
    default:
        break;
    }
    return NEURALNET_ERROR_BASE;
}
