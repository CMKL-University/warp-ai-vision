import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Computer Vision Apps',
    Svg: require('@site/static/img/undraw_inspiration_re_ivlv.svg').default,
    description: (
      <>
        Learn about applications of computer vision. How you can change the world, one app at a time!
      </>
    ),
  },
  {
    title: 'Basic Computer Vision',
    Svg: require('@site/static/img/undraw_optimize_image_re_3tb1.svg').default,
    description: (
      <>
        Fundamental knowledges of computer vision. Bitmap, filters and all the cool things you can do with images.
      </>
    ),
  },
  {
    title: 'Introduction to Neural Network',
    Svg: require('@site/static/img/undraw_mind_map_re_nlb6.svg').default,
    description: (
      <>
        Learn about artificial neural network and how it can be used to solve simple classification problem.
      </>
    ),
  },
  {
    title: 'CNN and Object Detection',
    Svg: require('@site/static/img/undraw_image_viewer_re_7ejc.svg').default,
    description: (
      <>
        The breakthrough model that sparked modern AI revolution.
        Convolutional neural network is most commonly applied to computer vision
        and helps computer see with minimal manual intervention.
      </>
    ),
  },
  {
    title: 'Seeing AI',
    Svg: require('@site/static/img/undraw_sentiment_analysis_jp6w.svg').default,
    description: (
      <>
        Build your own AI applications that can see the world in different ways.
      </>
    ),
  },
];

function Feature({Svg, jpg, title, description}) {
  if (jpg === undefined) {
    return (
      <div className={clsx('col col--4')}>
        <div className="text--center">
          <Svg className={styles.featureSvg} role="img" />
        </div>
        <div className="text--center padding-horiz--md">
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
      </div>
    );  
  } else
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <img className={styles.featureSvg} src={jpg}></img>
      </div>
      <div className="text--center padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
