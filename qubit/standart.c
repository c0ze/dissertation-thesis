

void error(char * message)
{
	printf("%s\n",message);
	exit(1);
}

int power (int base, int power)
{
	int i, result=1;
	for (i=0; i<power; i++)
		result *= base;
	return result;
}

__complex__ Complex_Multiplication (__complex__ a, __complex__ b)
{
	__complex__ double result;
	__real__ result = (__real__ a * __real__ b - __imag__ a * __imag__ b);
	__imag__ result = (__real__ a * __imag__ b + __imag__ a * __real__ b);
	return result ;
}

int power_of_two (int size)
{
	int i, counter = 0;
	for (i = 0; i < sizeof(int) * 8; i++)
		if ((size >> i) & 1)
			counter++;
		if (counter != 1)
			return 0;
		else
			return 1;
}

int get_biggest_2s_power (int size)
{
	int i;
	for (i=size; i>0; i--)
		if (power_of_two(i))
			return i;
}
