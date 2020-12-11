# FunDiversity
A repository for Functional Diversity Análisis
This is a set of code that can help you to calculate, filter and plot some trait data for functional diversity analysis
## Content
1. Filter data 
>This code works with an ordinary dataset with the following data columns: 
> - Individual trait measurements/observations (it should be numerical, not categorical)
> - Species data, that identifies the individuals
> - Abundance data
> - Treatment information that classifies the species in different groups
2. Cook traits
>It is often useful to calculate some traits that comes from field measurements. Traits like SAL (Specific leaf area), WD (Wood Density), LDMC (Leaf Dry Matter content) come from simple division of dry weight and fresh weight. Here is the code to calculate these traits.
3. Get and plot PCA 
> Working with a multitrait datasets needs to convert all these variables to a bidimensional space. Through the PCA it can be defined the functional niche of the species.
>
>This code get the PCA from the trait matrix and plot them with the relation of the variables in the two first components.
4. Calcualate Funcitonal Diversity indices
> Functional diversity from the multitrait perspective needs complex calculation for diversity mesurments. Here it define some indices, like Functional Richness (FRich) and plot the Convex Hull of the resulting comunity
5. Plot the Trait Probability Densitiy
> Abundance is important for diversity beacuase it recognices the strcture of the community in the ecosistem. Trough the Kernel Densitiy Models we can get the distribution of the species in the multitrait space weighted by their abundance in de comunity
>
>Here is made a plot based on this Trait Probability Density. Other version of this code will incorporate this perspective to the Functional diversity indices
