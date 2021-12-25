from torchvision.datasets import CocoDetection


def sortImgs(dataset):
    imgs = dataset.imgs
    targets = dataset.targets
    start_inds = [0]
    imgs_new = []
    targets_new = []
    cls = 0

    for i in range(len(targets) - 1):
        if targets[i] != targets[i + 1]:
            start_inds.append(i + 1)

    start_inds.append(len(targets))

    for i in range(0, len(start_inds) - 1, 2):
        if imgs[start_inds[i]][0].split('/')[6][3:] != 'negative':
            targets_new.extend([cls] * (start_inds[i + 1] - start_inds[i]))
            imgs_new.extend([(img[0], cls) for img in imgs[start_inds[i]: start_inds[i + 1]]])
            cls += 1
            targets_new.extend([cls] * (start_inds[i + 2] - start_inds[i + 1]))
            imgs_new.extend([(img[0], cls) for img in imgs[start_inds[i + 1]: start_inds[i + 2]]])
            cls += 1
        else:
            targets_new.extend([cls] * (start_inds[i + 2] - start_inds[i + 1]))
            imgs_new.extend([(img[0], cls) for img in imgs[start_inds[i + 1]: start_inds[i + 2]]])
            cls += 1
            targets_new.extend([cls] * (start_inds[i + 1] - start_inds[i]))
            imgs_new.extend([(img[0], cls) for img in imgs[start_inds[i]: start_inds[i + 1]]])
            cls += 1

    dataset.samples = imgs_new
    dataset.imgs = imgs_new
    dataset.targets = targets_new


def summarizeSuperCat(dataset: CocoDetection):
    superCat2CatId = {}
    PN2CatId = {
        'positive': [],
        'negative': [],
    }
    direction2SuperCat = {
        'cw': [],
        'ccw': [],
    }
    cats = dataset.coco.loadCats(dataset.coco.getCatIds())
    for cat in cats:
        catId = cat['id']
        catName = cat['name']
        superCat = cat['supercategory']
        direction = cat['direction']
        if superCat not in superCat2CatId:
            superCat2CatId[superCat] = [catId]
        else:
            superCat2CatId[superCat].append(catId)

        if 'negative' in catName:
            PN2CatId['negative'].append(catId)
        else:
            PN2CatId['positive'].append(catId)

        if superCat not in direction2SuperCat[direction]:
            direction2SuperCat[direction].append(superCat)

        summary = {
            'superCat2CatId': superCat2CatId,
            'PN2CatId': PN2CatId,
            'direction2SuperCat': direction2SuperCat,
        }
    return summary
