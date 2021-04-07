<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>جستجو</title>
    <link rel="stylesheet" href="css/style.css">
    <link rel="stylesheet" href="css/globalstyle.css">
</head>
<body>
<ul>
    <li><a href="#"> صفحه اصلی</a></li>
    <li><a href="php/register.php">صفحه ثبت نام</a></li>
    <li><a href="php/login.php">صفحه لاگین</a></li>
</ul>

<form accept-charset="UTF-8" action="php/results.php" method="post">
    <img src="image/google.jpg"><br>
    <input id="i1" type="text" name="search" required><br>
    <input type="radio" name="searchTitle" value="عنوان" checked> عنوان
    <input type="radio" name="searchTitle" value="عنوان و متن"> عنوان و متن
    <input id="i2" type="submit" value="جستجو">
</form>
</body>
</html>