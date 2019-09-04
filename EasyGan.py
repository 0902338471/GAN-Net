from numpy.random import rand
from numpy import  hstack
from numpy import zeros
from numpy import ones
from matplotlib import pyplot  
from keras.models import Sequential
from keras.layers import Dense,Activation
def generate_2d_points(size):
    x1=rand(size)-0.5
    x2=x1*x1
    x1=x1.reshape(size,1)
    x2=x2.reshape(size,1)
    return hstack((x1,x2))
def calculate(x):
    return x*x*x
def generate_real_data(size):
    x1=rand(size)-0.5
    x2=x1*x1
    x1=x1.reshape(size,1)
    x2=x2.reshape(size,1)
    y=ones((size,1))
    return hstack((x1,x2)),y
def define_discriminator(n_dimension=2):
    model=Sequential()
    model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_dimension))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
def generate_fake_data(size):
    x1=rand(size)-0.5
    x2=rand(size)-0.5
    x1=x1.reshape(size,1)
    x2=x2.reshape(size,1)
    y=zeros((size,1))
    return hstack((x1,x2)),y
def train_discriminator(model,n_epochs=1000,n_batch=128):
    for i in range(n_epochs):
        x_real,y_real=generate_real_data(n_batch)
        model.train_on_batch(x_real,y_real)
        x_fake,y_fake=generate_fake_data(n_batch)
        model.train_on_batch(x_fake,y_fake)
        _,acc_real=model.evaluate(x_real,y_real,verbose=0)
        _,acc_fake=model.evaluate(x_fake,y_fake,verbose=0)
        print(i,acc_real,acc_fake)

def define_generator(latent_dimension,n_dimension=2):
    model=Sequential()
    model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dimension))
    model.add(Dense(n_dimension,activation='linear'))
    return model
def generate_points_in_latent_space(latent_dimension,n_samples):
    x_input=rand(latent_dimension*n_samples)-0.5
    x_input=x_input.reshape(n_samples,latent_dimension)
    return x_input
def generate_real_samples(size):
    x1=rand(size)-0.5
    x2=x1*x1
    x1=x1.reshape(size,1)
    x2=x2.reshape(size,1)
    y=ones((size,1))
    return hstack((x1,x2)),y
def generate_fake_samples_for_visualizing(generator,latent_dimension,n):
    x_input=generate_points_in_latent_space(latent_dimension,n)
    x=generator.predict(x_input)
    pyplot.scatter(x[:,0],x[:,1])
    pyplot.show()
def generate_fake_samples(generator,latent_dimension,n):
    x_input=generate_points_in_latent_space(latent_dimension,n)
    x=generator.predict(x_input)
    y = zeros((n, 1))
    return x,y
def define_gan(generator,discriminator):
    discriminator.trainable=False
    model=Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy',optimizer='adam')
    return model
def summarize_performance(epoch,generator,discriminator,latent_dimension,n=100):
    x_real,y_real=generate_real_samples(n)
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)
    x_fake,y_fake=generate_fake_samples(generator,latent_dimension,n)
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)
    print(epoch, acc_real, acc_fake)
    pyplot.scatter(x_real[:,0],x_real[:,1],color='red')
    pyplot.scatter(x_fake[:,0],x_fake[:,1],color='blue')
    pyplot.show()
def gan_train(generator_model, discriminator_model, gan_model, latent_dim, n_epochs=10000, n_batch=128, n_eval=2000):
	for i in range(n_epochs):
		x_real, y_real = generate_real_samples(n_batch)
		x_fake, y_fake = generate_fake_samples(generator_model, latent_dim, n_batch)
		discriminator_model.train_on_batch(x_real, y_real)
		discriminator_model.train_on_batch(x_fake, y_fake)
		x_gan = generate_points_in_latent_space(latent_dim, n_batch)
		y_gan = ones((n_batch, 1))
		gan_model.train_on_batch(x_gan, y_gan)
		if (i+1) % n_eval == 0:
			summarize_performance(i, generator_model, discriminator_model, latent_dim)
#check point 1:generating two dimensional points
'''data = generate_2d_points(100)
pyplot.scatter(data[:, 0], data[:, 1])
pyplot.show()
'''
#check point 2:testing discriminator 
'''model=define_discriminator()
train_discriminator(model)
'''
#check point 3:generating 100 points from latent space
'''latent_dimension=5
model=define_generator(latent_dimension)
generate_fake_samples_for_visualizing(model,latent_dimension,100)
'''
#check point 4:train gan
latent_dimension=5
discriminator=define_discriminator()
generator=define_generator(latent_dimension)
gan_model=define_gan(generator,discriminator)
gan_train(generator,discriminator,gan_model,latent_dimension)
        
