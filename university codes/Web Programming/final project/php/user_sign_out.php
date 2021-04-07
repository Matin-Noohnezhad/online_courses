<?php

$string = '<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>ورود به حساب کاربری</title>
    <link rel="stylesheet" href="../css/userSignOut.css">
</head>
<body>
<form>
    <fieldset>
        <br>
        آیا مایل به خروج از حساب کاربری خود هستید؟
        <br>
        <br>
        <br>
        <br>
        <button id="yes" type="button" onclick="sign_out(true)">بله</button>
        <button id="no" type="button" onclick="sign_out(false)">خیر</button>
        <br>
        <br>
    </fieldset>
</form>

<script src="../js/userSignOut.js"></script>
</body>
</html>';

echo $string;

?>
